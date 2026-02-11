from pulp import *
import pandas as pd
import numpy as np

class InventoryOptimizer:
    def __init__(self, forecast_df, features_df):
        """
        Initialize the optimizer with forecast data and feature data (containing constraints).
        """
        self.forecast_df = forecast_df
        self.features_df = features_df
        
    def _prepare_data(self, store_id):
        """
        Prepares data for a specific store.
        """
        # Filter forecasts for this store
        # Assuming item_id format "Store_Sku" or we have a store_id column.
        # Based on previous steps, series_id = "Store_Sku".
        
        # We need to parse store_id from series_id if not present
        if "store_id" not in self.forecast_df.columns:
            self.forecast_df["store_id"] = self.forecast_df["item_id"].apply(lambda x: x.split("_")[0])
            
        store_forecast = self.forecast_df[self.forecast_df["store_id"] == store_id].copy()
        
        # Merge with features to get constraints
        # Features might be at (date, series_id) level
        # We need: unit_cost, holding_cost, spoilage_rate, moq, lead_time, price, etc.
        # These columns are in features.csv.
        
        # We'll merge on ['date', 'series_id'] (renaming item_id to series_id for merge)
        store_forecast["series_id"] = store_forecast["item_id"]
        store_forecast["date"] = pd.to_datetime(store_forecast["date"])
        
        # We need to make sure features_df has the same types
        feat_subset = self.features_df.copy()
        feat_subset["date"] = pd.to_datetime(feat_subset["date"])
        
        # Merge
        data = pd.merge(store_forecast, feat_subset, on=["date", "series_id"], how="left", suffixes=("", "_feat"))
        
        # Fill missing constraints with defaults
        data["moq"] = data["moq"].fillna(1)
        data["case_pack"] = data["case_pack"].fillna(1)
        data["lead_time_weeks"] = data["lead_time_weeks"].fillna(0).astype(int)
        data["unit_cost"] = data["unit_cost"].fillna(0)
        data["holding_cost_per_unit_week"] = data["holding_cost_per_unit_week"].fillna(0)
        data["sell_price"] = data["sell_price"].fillna(0) # Ensure price is not NaN
        
        data["stockout_penalty"] = data["sell_price"] * 0.5 
        # Fallback if price was 0 or missing
        data["stockout_penalty"] = data["stockout_penalty"].fillna(0)
        
        if "spoilage_rate_per_week" not in data.columns:
            data["spoilage_rate_per_week"] = 0.0
        else:
            data["spoilage_rate_per_week"] = data["spoilage_rate_per_week"].fillna(0)
            
        if "capacity_units" not in data.columns:
            data["capacity_units"] = 999999
        else:
            data["capacity_units"] = data["capacity_units"].fillna(999999) # Default infinite
            
        if "on_hand_start" not in data.columns:
            # Critical: If not present, we assume 0
            data["on_hand_start"] = 0.0
        else:
            data["on_hand_start"] = data["on_hand_start"].fillna(0.0)

        # Ensure forecasts are valid numbers
        if "mean_prediction" in data.columns:
             data["mean_prediction"] = data["mean_prediction"].fillna(0)
             
        # Check for any remaining NaNs in utilized columns and warn/fill
        cols_to_check = ["mean_prediction", "unit_cost", "holding_cost_per_unit_week", "stockout_penalty", "spoilage_rate_per_week", "on_hand_start"]
        for col in cols_to_check:
            if col in data.columns and data[col].isnull().any():
                print(f"[WARNING] NaNs detected in {col} for store {store_id}. Filling with 0.")
                data[col] = data[col].fillna(0)

        # Replace infs
        data = data.replace([np.inf, -np.inf], 0)

        return data

    def optimize_store(self, store_id, weeks=4):
        """
        Solves the inventory optimization problem for a single store for the next 'weeks'.
        """
        print(f"Optimizing inventory for Store: {store_id}...")
        
        data = self._prepare_data(store_id)
        
        # Unique items and dates
        items = data["series_id"].unique()
        dates = sorted(data["date"].unique())[:weeks] # optimization horizon
        
        # Create a mapping for easy access
        # dict[item][t] -> attributes
        params = {}
        for item in items:
            item_data = data[data["series_id"] == item].set_index("date")
            params[item] = item_data.to_dict('index')
            
        # Decision Variables
        # X[i, t]: Orders placed in week t
        # I[i, t]: Inventory at end of week t
        # L[i, t]: Lost sales in week t
        # Y[i, t]: Binary, 1 if order placed (for MOQ)
        
        vars_X = {}
        vars_I = {}
        vars_L = {}
        vars_Y = {}
        
        prob = LpProblem(f"Inventory_Optimization_{store_id}", LpMinimize)
        
        objective_terms = []
        
        for item in items:
            # Assume initial inventory is current on_hand? 
            # We need checking if 'on_hand_start' is in features for the first week.
            first_date = dates[0]
            initial_inventory = params[item][first_date].get("on_hand_start", 0)
            
            prev_inv_var = None
            
            for t_idx, date in enumerate(dates):
                # Parameters
                p = params[item][date]
                forecast = p.get("mean_prediction", 0) # Use mean forecast
                lead_time = int(p.get("lead_time_weeks", 0))
                moq = p.get("moq", 1)
                pack_size = p.get("case_pack", 1)
                
                # Costs
                unit_cost = p.get("unit_cost", 10.0) # Default 10 if missing
                holding_cost = p.get("holding_cost_per_unit_week", 1.0)
                
                # Stockout Penalty
                # If sell_price is missing (0), use a high default relative to cost
                price = p.get("sell_price", 0)
                if price <= 0:
                    stockout_cost = unit_cost * 2 # Default penalty: 2x cost involved
                else:
                    stockout_cost = price * 1.5 # Penalty: 1.5x Price (Lost margin + goodwill)
                
                spoilage_rate = p.get("spoilage_rate_per_week", 0)
                
                # Variables
                # Orders are in Packs! X_packs
                x_packs = LpVariable(f"OrderPacks_{item}_{t_idx}", 0, None, LpInteger)
                x_units = x_packs * pack_size
                
                inv = LpVariable(f"Inv_{item}_{t_idx}", 0, None, LpContinuous)
                lost = LpVariable(f"Lost_{item}_{t_idx}", 0, None, LpContinuous)
                is_ord = LpVariable(f"IsOrd_{item}_{t_idx}", 0, 1, LpBinary)
                
                vars_X[(item, t_idx)] = x_units
                vars_I[(item, t_idx)] = inv
                vars_L[(item, t_idx)] = lost
                vars_Y[(item, t_idx)] = is_ord
                
                # --- Constraints ---
                
                # 1. Flow Balance
                # Inv[t] = Inv[t-1] + Arrivals[t] - Sales[t] - Spoilage[t]
                # Arrivals[t] come from orders placed at t - lead_time
                
                order_arrival = 0
                if t_idx >= lead_time:
                    # Order placed 'lead_time' weeks ago arrives now
                    # We need the variable from that time
                     order_arrival = vars_X[(item, t_idx - lead_time)]
                else:
                    # Orders placed before horizon? Assume 0 for now or user input pipeline
                    order_arrival = 0 
                
                if t_idx == 0:
                    prev_inv = initial_inventory
                else:
                    prev_inv = vars_I[(item, t_idx - 1)]
                    
                # Spoilage is fraction of ending inventory? Or beginning? 
                # Let's simplify: Spoilage reduces available inventory for NEXT period.
                # Balance:
                # Inv[t] = (PrevInv + Arrival) - (Forecast - Lost)
                # Note: (Forecast - Lost) is "Met Demand"
                
                prob += inv == prev_inv + order_arrival - (forecast - lost)
                
                # 2. Lost Sales cannot exceed Forecast
                prob += lost <= forecast
                
                # 3. MOQ constraint
                # X_units >= MOQ * IsOrdered
                # X_units <= BigM * IsOrdered
                BigM = 100000 
                prob += x_units >= moq * is_ord
                prob += x_units <= BigM * is_ord
                
                # 4. Safety Stock Constraint
                # Force ending inventory to cover some fraction of future demand?
                # Or simple minimum level.
                # Let's use 50% of current week's demand as proxy for variability coverage
                # (Ideally use forecast variance, but mean is available)
                safety_stock = forecast * 0.5
                prob += inv >= safety_stock
                
                # --- Objective ---
                # Minimize: Ordering Cost (Unit Cost * Units) + Holding + Stockout + Spoilage
                # Note: Spoilage cost roughly modeled as % of inventory lost * unit cost
                 
                obj_t = (x_units * unit_cost) + \
                        (inv * holding_cost) + \
                        (lost * stockout_cost) + \
                        (inv * spoilage_rate * unit_cost)
                        
                objective_terms.append(obj_t)
                
        # 4. Capacity Constraint (Per Store, Per Week)
        for t_idx, date in enumerate(dates):
            # Sum of inventory of all items in this store at week t <= Capacity
            # We assume capacity is constant or we take specific week's capacity from any item record
            # (assuming identical for all items in store)
            
            # Using first item's capacity record for simplicity (data usually repeated)
            cap = params[items[0]][date].get("capacity_units", 999999)
            if cap <= 0: cap = 999999 
            
            total_inv_t = lpSum([vars_I[(item, t_idx)] for item in items])
            prob += total_inv_t <= cap
            
            # 5. Supplier Limit (Optional, Per Supplier Per Week)
            # We need to group items by supplier
            # ... (Skipping for now unless explicit supplier grouping requested in data)
            
        prob += lpSum(objective_terms)
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=False))
        print(f"Status: {LpStatus[prob.status]}")
        
        # Extract Results
        results = []
        for item in items:
            for t_idx, date in enumerate(dates):
                res = {
                    "store_id": store_id,
                    "item_id": item,
                    "date": date,
                    "order_units": value(vars_X[(item, t_idx)]),
                    "projected_inventory": value(vars_I[(item, t_idx)]),
                    "lost_sales": value(vars_L[(item, t_idx)]),
                    "forecast_mean": params[item][date].get("mean_prediction", 0)
                }
                results.append(res)
                
        return pd.DataFrame(results)

    def optimize_all(self, horizon_weeks=4):
        store_ids = self.forecast_df["item_id"].apply(lambda x: x.split("_")[0]).unique()
        all_results = []
        
        for store in store_ids:
            try:
                df_res = self.optimize_store(store, weeks=horizon_weeks)
                all_results.append(df_res)
            except Exception as e:
                print(f"Error optimizing store {store}: {e}")
                
        if not all_results:
            return pd.DataFrame()
            
        return pd.concat(all_results, ignore_index=True)
