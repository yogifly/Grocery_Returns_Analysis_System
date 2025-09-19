import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data():
    """Generate sample data for grocery returns analysis"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Product data
    products = [
        {'product_id': 'APPLE_001', 'product_name': 'Red Apples', 'category': 'fruit', 'expected_shelf_life_days': 7},
        {'product_id': 'BANANA_001', 'product_name': 'Bananas', 'category': 'fruit', 'expected_shelf_life_days': 5},
        {'product_id': 'AVOCADO_001', 'product_name': 'Avocados', 'category': 'fruit', 'expected_shelf_life_days': 4},
        {'product_id': 'SPINACH_001', 'product_name': 'Fresh Spinach', 'category': 'vegetable', 'expected_shelf_life_days': 3},
        {'product_id': 'TOMATO_001', 'product_name': 'Tomatoes', 'category': 'vegetable', 'expected_shelf_life_days': 6},
        {'product_id': 'LETTUCE_001', 'product_name': 'Lettuce', 'category': 'vegetable', 'expected_shelf_life_days': 4},
        {'product_id': 'BERRIES_001', 'product_name': 'Mixed Berries', 'category': 'fruit', 'expected_shelf_life_days': 3},
        {'product_id': 'CARROT_001', 'product_name': 'Carrots', 'category': 'vegetable', 'expected_shelf_life_days': 10},
    ]
    
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad', 'Ahmedabad']
    carriers = ['FastDelivery', 'QuickShip', 'ExpressLogistics', 'SpeedyDelivery']
    package_types = ['standard', 'insulated', 'premium']
    
    # Generate orders data
    orders_data = []
    returns_data = []
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 3, 1)
    
    order_id_counter = 1
    return_id_counter = 1
    
    for _ in range(5000):  # Generate 5000 orders
        order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        delivery_date = order_date + timedelta(days=random.randint(1, 3))
        
        product = random.choice(products)
        city = random.choice(cities)
        carrier = random.choice(carriers)
        package_type = random.choice(package_types)
        
        order = {
            'order_id': f'ORD_{order_id_counter:06d}',
            'customer_id': f'CUST_{random.randint(1, 1000):04d}',
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'quantity': random.randint(1, 5),
            'order_date': order_date,
            'delivery_date': delivery_date,
            'delivered_location': city,
            'carrier_id': carrier,
            'package_type': package_type,
            'batch_id': f'BATCH_{random.randint(1, 100):03d}'
        }
        
        orders_data.append(order)
        
        # Generate returns (20% return rate, higher for certain products/conditions)
        return_probability = 0.15
        
        # Increase return probability for certain conditions
        if product['product_id'] in ['AVOCADO_001', 'BERRIES_001', 'SPINACH_001']:
            return_probability = 0.25
        if city in ['Mumbai', 'Chennai']:  # Coastal cities with higher humidity
            return_probability += 0.05
        if carrier == 'QuickShip':  # One carrier has issues
            return_probability += 0.08
        if package_type == 'standard':
            return_probability += 0.03
            
        if random.random() < return_probability:
            # Generate return
            days_until_return = random.randint(1, min(product['expected_shelf_life_days'] + 2, 8))
            return_date = delivery_date + timedelta(days=days_until_return)
            
            # Return reasons with realistic distribution
            reason_weights = {
                'stale': 0.4,
                'damaged': 0.25,
                'quality_issues': 0.15,
                'wrong_item': 0.1,
                'packaging_issues': 0.05,
                'late_delivery': 0.05
            }
            
            return_reason = np.random.choice(list(reason_weights.keys()), p=list(reason_weights.values()))
            
            # Generate realistic return reason text
            reason_texts = {
                'stale': ['Product was stale', 'Items were rotten', 'Found mold on the product', 'Spoiled items received'],
                'damaged': ['Items were damaged', 'Bruised fruits', 'Packaging was damaged', 'Products were crushed'],
                'quality_issues': ['Poor quality', 'Not fresh', 'Quality below expectations', 'Items looked old'],
                'wrong_item': ['Wrong product delivered', 'Incorrect item', 'Different product than ordered'],
                'packaging_issues': ['Poor packaging', 'Inadequate packaging', 'Packaging leaked'],
                'late_delivery': ['Delivery was late', 'Delayed delivery', 'Received after expected date']
            }
            
            return_text = random.choice(reason_texts[return_reason])
            
            return_record = {
                'return_id': f'RET_{return_id_counter:06d}',
                'order_id': order['order_id'],
                'return_date': return_date,
                'return_reason': return_text,
                'return_reason_category': return_reason,
                'refund_amount': random.randint(50, 500)
            }
            
            returns_data.append(return_record)
            return_id_counter += 1
        
        order_id_counter += 1
    
    # Create DataFrames
    orders_df = pd.DataFrame(orders_data)
    returns_df = pd.DataFrame(returns_data)
    products_df = pd.DataFrame(products)
    
    return orders_df, returns_df, products_df

if __name__ == "__main__":
    orders_df, returns_df, products_df = generate_sample_data()
    
    # Save to CSV files
    orders_df.to_csv('data/orders.csv', index=False)
    returns_df.to_csv('data/returns.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)
    
    print("Sample data generated successfully!")
    print(f"Orders: {len(orders_df)} records")
    print(f"Returns: {len(returns_df)} records")
    print(f"Products: {len(products_df)} records")
