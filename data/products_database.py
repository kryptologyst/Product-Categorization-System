"""
Mock product database with diverse categories and realistic product data.
This module provides a comprehensive dataset for training the product categorization system.
"""

import pandas as pd
import random
from typing import List, Dict, Any

class ProductDatabase:
    """Mock database containing diverse product categories and descriptions."""
    
    def __init__(self):
        self.products_data = self._generate_product_data()
        self.df = pd.DataFrame(self.products_data)
    
    def _generate_product_data(self) -> Dict[str, List[str]]:
        """Generate comprehensive product data across multiple categories."""
        
        products = {
            'ProductName': [],
            'Description': [],
            'Category': [],
            'Price': [],
            'Brand': []
        }
        
        # Electronics products
        electronics = [
            ("iPhone 15 Pro Max 512GB Space Black", "Latest Apple smartphone with titanium design and advanced camera system", "Apple", 1199.99),
            ("Samsung Galaxy S24 Ultra 256GB", "Premium Android smartphone with S Pen and AI features", "Samsung", 1299.99),
            ("MacBook Pro 16-inch M3 Max", "Professional laptop with Apple Silicon for creative professionals", "Apple", 2499.99),
            ("Dell XPS 13 Plus Intel i7", "Ultra-portable Windows laptop with premium build quality", "Dell", 1399.99),
            ("Sony WH-1000XM5 Wireless Headphones", "Industry-leading noise canceling over-ear headphones", "Sony", 399.99),
            ("iPad Air 5th Generation 64GB", "Versatile tablet perfect for work and creativity", "Apple", 599.99),
            ("Nintendo Switch OLED Console", "Hybrid gaming console with vibrant OLED display", "Nintendo", 349.99),
            ("Canon EOS R6 Mark II Camera", "Full-frame mirrorless camera for photography enthusiasts", "Canon", 2499.99),
            ("Bose QuietComfort Earbuds", "Premium wireless earbuds with noise cancellation", "Bose", 279.99),
            ("LG OLED C3 55-inch 4K TV", "Premium OLED smart TV with perfect blacks and vibrant colors", "LG", 1499.99),
            ("Razer DeathAdder V3 Gaming Mouse", "High-precision gaming mouse with ergonomic design", "Razer", 99.99),
            ("Logitech MX Master 3S Wireless Mouse", "Advanced wireless mouse for productivity professionals", "Logitech", 99.99),
            ("Samsung 980 PRO 2TB NVMe SSD", "High-performance internal SSD for gaming and professional work", "Samsung", 199.99),
            ("ASUS ROG Strix RTX 4080 Graphics Card", "High-end graphics card for 4K gaming and content creation", "ASUS", 1199.99),
            ("Corsair K95 RGB Platinum Keyboard", "Premium mechanical gaming keyboard with RGB lighting", "Corsair", 199.99)
        ]
        
        # Clothing & Fashion
        clothing = [
            ("Levi's 501 Original Fit Jeans", "Classic straight-leg denim jeans in vintage wash", "Levi's", 89.99),
            ("Nike Air Force 1 Low White", "Iconic basketball sneakers in classic white colorway", "Nike", 110.00),
            ("Patagonia Better Sweater Fleece Jacket", "Sustainable fleece jacket perfect for outdoor activities", "Patagonia", 139.99),
            ("Ray-Ban Aviator Classic Sunglasses", "Timeless aviator sunglasses with premium lenses", "Ray-Ban", 154.99),
            ("Adidas Ultraboost 22 Running Shoes", "High-performance running shoes with responsive cushioning", "Adidas", 190.00),
            ("The North Face Venture 2 Rain Jacket", "Waterproof and breathable jacket for outdoor adventures", "The North Face", 99.99),
            ("Calvin Klein Cotton Classic Fit T-Shirt", "Premium cotton t-shirt with comfortable fit", "Calvin Klein", 29.99),
            ("Converse Chuck Taylor All Star High Top", "Classic canvas sneakers in timeless high-top design", "Converse", 65.00),
            ("Uniqlo Heattech Ultra Warm Crew Neck Long Sleeve T-Shirt", "Advanced thermal underwear for cold weather", "Uniqlo", 19.90),
            ("Zara Wool Blend Overcoat", "Elegant wool coat perfect for formal occasions", "Zara", 199.99)
        ]
        
        # Home & Garden
        home_garden = [
            ("KitchenAid Artisan Stand Mixer 5-Quart", "Professional-grade stand mixer for baking enthusiasts", "KitchenAid", 379.99),
            ("Dyson V15 Detect Cordless Vacuum", "Advanced cordless vacuum with laser dust detection", "Dyson", 749.99),
            ("Instant Pot Duo 7-in-1 Electric Pressure Cooker", "Multi-functional pressure cooker for quick meals", "Instant Pot", 99.95),
            ("Philips Hue White and Color Ambiance Smart Bulb", "Smart LED bulb with millions of colors and voice control", "Philips", 49.99),
            ("Ninja Foodi Personal Blender", "Compact blender perfect for smoothies and protein shakes", "Ninja", 79.99),
            ("Roomba i7+ Robot Vacuum", "Self-emptying robot vacuum with smart mapping", "iRobot", 599.99),
            ("Weber Genesis II E-315 Gas Grill", "Premium gas grill perfect for backyard barbecues", "Weber", 799.99),
            ("Casper Original Mattress Queen Size", "Premium foam mattress designed for optimal sleep comfort", "Casper", 1095.00),
            ("Nest Learning Thermostat 3rd Generation", "Smart thermostat that learns your schedule and saves energy", "Google", 249.99),
            ("IKEA HEMNES Daybed with 3 Drawers", "Versatile daybed with built-in storage solution", "IKEA", 329.00)
        ]
        
        # Books & Media
        books_media = [
            ("The Seven Husbands of Evelyn Hugo Novel", "Captivating fiction about a reclusive Hollywood icon", "Generic", 16.99),
            ("Atomic Habits by James Clear", "Practical guide to building good habits and breaking bad ones", "Generic", 18.99),
            ("The Thursday Murder Club Mystery Series", "Cozy mystery series featuring retirement home residents", "Generic", 15.99),
            ("Dune: Complete Series Box Set", "Epic science fiction saga in collector's edition", "Generic", 89.99),
            ("National Geographic Kids Almanac 2024", "Educational reference book packed with facts and photos", "National Geographic", 12.99),
            ("The Art of War by Sun Tzu", "Classic strategy guide with modern applications", "Generic", 9.99),
            ("Becoming by Michelle Obama Audiobook", "Inspiring memoir narrated by the former First Lady", "Generic", 24.99),
            ("Marvel Comics Spider-Man Omnibus", "Comprehensive collection of classic Spider-Man comics", "Marvel", 125.00),
            ("The Great Gatsby Leather Bound Edition", "Classic American literature in premium binding", "Generic", 34.99),
            ("Programming Pearls 2nd Edition", "Essential computer science and programming concepts", "Generic", 49.99)
        ]
        
        # Sports & Outdoors
        sports_outdoors = [
            ("Yeti Rambler 30 oz Tumbler", "Insulated stainless steel tumbler for hot and cold drinks", "Yeti", 39.99),
            ("Coleman Sundome 4-Person Tent", "Reliable camping tent with easy setup and weather protection", "Coleman", 89.99),
            ("Hydro Flask 32 oz Wide Mouth Water Bottle", "Insulated water bottle perfect for outdoor activities", "Hydro Flask", 44.95),
            ("REI Co-op Trail 40 Backpack", "Versatile hiking backpack with comfortable suspension system", "REI", 149.99),
            ("Patagonia Houdini Windbreaker Jacket", "Ultra-lightweight packable jacket for outdoor adventures", "Patagonia", 129.99),
            ("Black Diamond Spot 400 Headlamp", "Bright and reliable headlamp for camping and hiking", "Black Diamond", 49.95),
            ("Osprey Daylite Plus Daypack", "Comfortable daypack perfect for hiking and daily use", "Osprey", 65.00),
            ("Merrell Moab 3 Hiking Boots", "Durable and comfortable hiking boots for all terrains", "Merrell", 139.99),
            ("Therm-a-Rest NeoAir XLite Sleeping Pad", "Ultralight sleeping pad for backpacking adventures", "Therm-a-Rest", 199.95),
            ("Goal Zero Nomad 20 Solar Panel", "Portable solar panel for charging devices outdoors", "Goal Zero", 199.95)
        ]
        
        # Health & Beauty
        health_beauty = [
            ("Olaplex No. 3 Hair Perfector Treatment", "Professional hair treatment for damaged and chemically treated hair", "Olaplex", 28.00),
            ("CeraVe Daily Moisturizing Lotion", "Gentle moisturizer suitable for normal to dry skin", "CeraVe", 16.99),
            ("The Ordinary Niacinamide 10% + Zinc 1%", "Concentrated serum to reduce appearance of blemishes", "The Ordinary", 7.90),
            ("Philips Sonicare DiamondClean Electric Toothbrush", "Premium electric toothbrush with multiple cleaning modes", "Philips", 199.99),
            ("Neutrogena Ultra Sheer Dry-Touch Sunscreen SPF 100+", "High-protection sunscreen with non-greasy formula", "Neutrogena", 11.99),
            ("Fenty Beauty Pro Filt'r Soft Matte Foundation", "Long-wearing foundation with buildable coverage", "Fenty Beauty", 39.00),
            ("Glossier Cloud Paint Gel Blush", "Buildable gel blush for a natural flush of color", "Glossier", 20.00),
            ("Drunk Elephant C-Firma Day Serum", "Antioxidant serum with vitamin C for brighter skin", "Drunk Elephant", 80.00),
            ("Laneige Lip Sleeping Mask", "Overnight lip treatment for soft and smooth lips", "Laneige", 24.00),
            ("Foreo Luna 3 Facial Cleansing Brush", "Smart facial cleansing device with personalized routines", "Foreo", 199.00)
        ]
        
        # Automotive
        automotive = [
            ("Michelin Pilot Sport 4S Tire 245/40R18", "High-performance tire for sports cars and luxury vehicles", "Michelin", 289.99),
            ("Chemical Guys Complete Car Care Kit", "Professional-grade car detailing products and accessories", "Chemical Guys", 149.99),
            ("Garmin DriveSmart 65 GPS Navigator", "Advanced GPS with voice-activated navigation and traffic updates", "Garmin", 199.99),
            ("WeatherTech All-Weather Floor Mats", "Custom-fit floor protection for all weather conditions", "WeatherTech", 159.95),
            ("Thule Motion XT Rooftop Cargo Box", "Aerodynamic cargo box for additional storage capacity", "Thule", 649.95),
            ("Anker Roav DashCam C1 Pro", "Compact dashboard camera with night vision and app connectivity", "Anker", 69.99),
            ("Rain-X Latitude Water Repellency Wiper Blades", "Premium wiper blades with water-repelling technology", "Rain-X", 24.99),
            ("Armor All Car Vacuum Cleaner", "Portable vacuum designed specifically for automotive cleaning", "Armor All", 79.99),
            ("Covercraft Custom Fit Car Cover", "Weather-resistant car cover with custom fit guarantee", "Covercraft", 199.99),
            ("NOCO Genius G3500 Battery Charger", "Smart battery charger and maintainer for 6V and 12V batteries", "NOCO", 79.95)
        ]
        
        # Add all categories to the main products dictionary
        all_categories = [
            (electronics, "Electronics"),
            (clothing, "Clothing & Fashion"),
            (home_garden, "Home & Garden"),
            (books_media, "Books & Media"),
            (sports_outdoors, "Sports & Outdoors"),
            (health_beauty, "Health & Beauty"),
            (automotive, "Automotive")
        ]
        
        for category_products, category_name in all_categories:
            for product_name, description, brand, price in category_products:
                products['ProductName'].append(product_name)
                products['Description'].append(description)
                products['Category'].append(category_name)
                products['Brand'].append(brand)
                products['Price'].append(price)
        
        return products
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the complete product database as a pandas DataFrame."""
        return self.df.copy()
    
    def get_categories(self) -> List[str]:
        """Return list of unique product categories."""
        return self.df['Category'].unique().tolist()
    
    def get_products_by_category(self, category: str) -> pd.DataFrame:
        """Return products filtered by category."""
        return self.df[self.df['Category'] == category].copy()
    
    def search_products(self, query: str) -> pd.DataFrame:
        """Search products by name or description."""
        mask = (self.df['ProductName'].str.contains(query, case=False, na=False) |
                self.df['Description'].str.contains(query, case=False, na=False))
        return self.df[mask].copy()
    
    def add_product(self, name: str, description: str, category: str, brand: str, price: float):
        """Add a new product to the database."""
        new_product = {
            'ProductName': name,
            'Description': description,
            'Category': category,
            'Brand': brand,
            'Price': price
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_product])], ignore_index=True)
    
    def get_sample_data(self, n: int = 50) -> pd.DataFrame:
        """Return a random sample of n products."""
        return self.df.sample(n=min(n, len(self.df))).copy()

if __name__ == "__main__":
    # Test the database
    db = ProductDatabase()
    print(f"Total products: {len(db.df)}")
    print(f"Categories: {db.get_categories()}")
    print("\nSample products:")
    print(db.get_sample_data(5)[['ProductName', 'Category', 'Price']])
