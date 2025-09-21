#!/usr/bin/env python3
from dotenv import load_dotenv
from nicegui import ui
import router
import os

load_dotenv()

# Ladda din egen router-frame (i stället för den inbyggda)
ui.add_head_html('<script type="module" src="router_frame.js"></script>')

# SPA layout
with ui.element('router-frame').props('default-page=/'):
    pass

ui.run(
    storage_secret=os.getenv('STORAGE_SECRET'),
    reload=os.getenv('RELOAD', 'false').lower() == 'true',
    port=int(os.getenv('PORT', 8080))
)

