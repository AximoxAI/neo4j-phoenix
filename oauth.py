import requests
from kiteconnect import KiteConnect
import webbrowser
from urllib.parse import urlparse, parse_qs

def generate_access_token():
    """
    Helper script to generate Kite Connect access token
    """
    
    # Your app credentials
    API_KEY = "14u2ycp8m5rjm81e"  # Replace with your API key
    API_SECRET = "mavzkd6me8b58ixhw90hoje1jn0tm52x"  # Replace with your API secret
    # Step 1: Create KiteConnect instance
    kite = KiteConnect(api_key=API_KEY)
    
    # Step 2: Get login URL
    login_url = kite.login_url()
    print(f"🔗 Please visit this URL to login:")
    print(login_url)
    print()
    
    # Automatically open browser (optional)
    webbrowser.open(login_url)
    
    # Step 3: Get request token from redirect URL
    print("📝 After logging in, you'll be redirected to a URL like:")
    print("https://127.0.0.1:8080/?request_token=XXXXXX&action=login&status=success")
    print()
    
    # Get request token from user
    redirect_url = input("🔗 Paste the complete redirect URL here: ")
    
    # Parse request token from URL
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)
    request_token = query_params.get('request_token', [None])[0]
    
    if not request_token:
        print("❌ Could not find request token in URL!")
        return None
    
    print(f"✅ Found request token: {request_token}")
    
    try:
        # Step 4: Generate access token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        
        print(f"\n🎉 Success! Your access token is:")
        print(f"📋 {access_token}")
        print(f"\n💾 Save this token - it's valid until market close.")
        
        # Test the token
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"\n✅ Token verified! Logged in as: {profile['user_name']}")
        
        return {
            "access_token": access_token,
            "user_id": data.get("user_id"),
            "user_name": profile.get("user_name")
        }
        
    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Kite Connect Access Token Generator")
    print("=" * 40)
    
    # Instructions
    print("\n📚 Before running this script:")
    print("1. Replace API_KEY and API_SECRET with your actual values")
    print("2. Make sure your Kite Connect app is approved")
    print("3. Ensure your redirect URL is set to https://127.0.0.1:8080")
    print("\n" + "=" * 40)
    
    # Generate token
    result = generate_access_token()
    
    if result:
        print(f"\n📝 Use these credentials in your trading app:")
        print(f"API Key: {API_KEY}")
        print(f"API Secret: {API_SECRET}")
        print(f"Access Token: {result['access_token']}")