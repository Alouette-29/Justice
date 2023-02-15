from flask import Flask, render_template
import requests
import json

app = Flask(__name__)

def get_meme():
    