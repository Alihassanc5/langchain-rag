import hashlib
import hmac
import streamlit as st
from config import ADMIN_PASSWORD


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password: str, hashed_password: str) -> bool:
    """Check if a password matches the hashed password."""
    return hmac.compare_digest(hash_password(password), hashed_password)


def init_authentication():
    """Initialize authentication settings."""
    # Set default admin credentials if not already set
    if "admin_password_hash" not in st.session_state:
        st.session_state.admin_password_hash = hash_password(ADMIN_PASSWORD)
    
    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False


def authenticate_user():
    """Authenticate user for admin access."""
    if not st.session_state.authenticated:
        st.sidebar.header("🔐 Admin Authentication")
        
        # Password input
        password = st.sidebar.text_input(
            "Enter Admin Password",
            type="password",
            help="Enter the admin password to access document management features"
        )
        
        # Login button
        if st.sidebar.button("🔑 Login"):
            if check_password(password, st.session_state.admin_password_hash):
                st.session_state.authenticated = True
                st.sidebar.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid password!")
                
        return False
    
    return True
