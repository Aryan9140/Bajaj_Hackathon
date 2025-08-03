# fix_python312.py - Python 3.12 Compatibility Fix
import os
import sys
import shutil

def backup_current_main():
    """Backup current main.py"""
    main_path = "app/main.py"
    backup_path = "app/main_full.py"
    
    if os.path.exists(main_path):
        shutil.copy2(main_path, backup_path)
        print(f"✅ Backed up current main.py to main_full.py")
    else:
        print(f"⚠️ No main.py found to backup")

def use_simple_version():
    """Switch to Python 3.12 compatible simple version"""
    simple_path = "app/main_simple.py"
    main_path = "app/main.py"
    
    if os.path.exists(simple_path):
        shutil.copy2(simple_path, main_path)
        print(f"✅ Switched to Python 3.12 compatible version")
        print(f"📄 main.py is now using simplified RAG system")
    else:
        print(f"❌ main_simple.py not found")
        return False
    
    return True

def use_full_version():
    """Switch back to full version"""
    full_path = "app/main_full.py"
    main_path = "app/main.py"
    
    if os.path.exists(full_path):
        shutil.copy2(full_path, main_path)
        print(f"✅ Switched back to full RAG system")
        print(f"📄 main.py is now using full enhanced version")
    else:
        print(f"❌ main_full.py backup not found")
        return False
    
    return True

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print(f"⚠️ Python 3.12+ detected - may have compatibility issues with some LangChain versions")
        return True
    else:
        print(f"✅ Python version should be compatible")
        return False

def install_compatible_packages():
    """Install Python 3.12 compatible packages"""
    print("📦 Installing Python 3.12 compatible packages...")
    
    commands = [
        "pip install --upgrade pydantic==2.5.3",
        "pip install --upgrade langchain==0.1.9",
        "pip install --upgrade langchain-core==0.1.23",
        "pip install --upgrade langchain-community==0.0.20",
        "pip install --upgrade langchain-openai==0.0.8",
        "pip install --upgrade openai==1.12.0",
        "pip install --upgrade typing-extensions==4.9.0"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        os.system(cmd)
    
    print("✅ Package installation completed")

def main():
    """Main fix function"""
    print("🔧 Python 3.12 Compatibility Fix Tool")
    print("=" * 50)
    
    # Check Python version
    is_python312 = check_python_version()
    
    print("\nChoose an option:")
    print("1. Use Simple Version (Python 3.12 compatible, basic RAG)")
    print("2. Use Full Version (Advanced RAG with potential compatibility issues)")
    print("3. Install Compatible Packages")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🔄 Switching to Simple Version...")
        backup_current_main()
        if use_simple_version():
            print("\n✅ SUCCESS!")
            print("🚀 Now run: python run.py")
            print("📝 Features available:")
            print("   • OpenAI GPT-4o for answers")
            print("   • PDF processing")
            print("   • Basic document chunking")
            print("   • Context matching")
            print("   • Python 3.12 compatible")
    
    elif choice == "2":
        print("\n🔄 Switching to Full Version...")
        if use_full_version():
            print("\n✅ SUCCESS!")
            print("⚠️ May have compatibility issues with Python 3.12")
            print("🚀 Try: python run.py")
        
    elif choice == "3":
        print("\n📦 Installing Compatible Packages...")
        install_compatible_packages()
        print("\n✅ Try running the server now")
    
    elif choice == "4":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()