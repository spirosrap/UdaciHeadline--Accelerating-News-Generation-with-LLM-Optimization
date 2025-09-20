#!/bin/bash

# Secure setup script for Hugging Face token
# This script helps you set up your Hugging Face token securely

echo "🔐 Hugging Face Token Setup"
echo "=========================="
echo ""
echo "This script will help you set up your Hugging Face token securely."
echo "Your token will be stored as an environment variable, not in the code."
echo ""

# Check if token is already set
if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "✅ Token is already set in environment variables"
    echo "Token starts with: ${HUGGINGFACE_HUB_TOKEN:0:10}..."
else
    echo "⚠️  No token found in environment variables"
    echo ""
    echo "To set your token, run one of these commands:"
    echo ""
    echo "For current session only:"
    echo "export HUGGINGFACE_HUB_TOKEN=your_token_here"
    echo ""
    echo "For permanent setup (add to ~/.bashrc):"
    echo "echo 'export HUGGINGFACE_HUB_TOKEN=your_token_here' >> ~/.bashrc"
    echo "source ~/.bashrc"
    echo ""
    echo "Replace 'your_token_here' with your actual Hugging Face token"
fi

echo ""
echo "🔍 Testing token access..."
if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "✅ Token is available for use"
else
    echo "❌ Token not set - please set it before running the notebook"
fi
