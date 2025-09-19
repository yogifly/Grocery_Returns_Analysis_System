import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_data_generator import generate_sample_data

# Generate sample data
orders_df, returns_df, products_df = generate_sample_data()

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save to CSV files
orders_df.to_csv('data/orders.csv', index=False)
returns_df.to_csv('data/returns.csv', index=False)
products_df.to_csv('data/products.csv', index=False)

print("✅ Sample data generated successfully!")
print(f"📦 Orders: {len(orders_df)} records")
print(f"🔄 Returns: {len(returns_df)} records")
print(f"🥬 Products: {len(products_df)} records")
print(f"📊 Return rate: {len(returns_df)/len(orders_df)*100:.1f}%")
