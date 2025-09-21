
from nicegui import ui, app
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
import os




# --- Färgkonstanter (styr via .env, fallback till default) ---
PRIMARY_COLOR = os.getenv('PRIMARY_COLOR', '#0B4271')
SECONDARY_COLOR = os.getenv('SECONDARY_COLOR', '#C77DFF')
BACKGROUND_COLOR = os.getenv('BACKGROUND_COLOR', '#F8F9FA')
TEXT_COLOR = os.getenv('TEXT_COLOR', '#212529')
TEXT_COLOR_LIGHT = os.getenv('TEXT_COLOR_LIGHT', '#EAEAEA')

app_name = os.getenv('APP_NAME')
css_prefix = (app_name or 'app').lower().replace(' ', '-')

# --- fill_main_header och fill_main_drawer funktionerna (utan user-data) ---
def fill_main_header(left_drawer_instance):
    with ui.header().classes(f'{css_prefix}-header items-center'):
        ui.button(on_click=lambda: left_drawer_instance.toggle(), icon='menu').props('flat color=white')
        ui.label(app_name).classes('text-xl font-bold ml-2')
        ui.space()

def fill_main_drawer(left_drawer_instance):
    with left_drawer_instance.classes(f'{css_prefix}-drawer'):
        # ui.label('Menu').classes('text-lg font-bold mb-2')
        ui.link('Dashboard', '/').classes(f'{css_prefix}-link my-2 text-lg')
        with ui.expansion('Kuponger').props('expand-separator').classes('my-2 text-lg'):
            ui.link('Visa alla kuponger', '/kuponger/list').classes(f'{css_prefix}-link ml-4 my-1 text-base')
            ui.link('Lägg till ny kupong', '/kuponger/add').classes(f'{css_prefix}-link ml-4 my-1 text-base')
            ui.link('Historik', '/kuponger/list').classes(f'{css_prefix}-link ml-4 my-1 text-base')
        with ui.expansion('Spela').props('expand-separator').classes('my-2 text-lg'):
            ui.link('Visa alla spel', '/spel/list').classes(f'{css_prefix}-link ml-4 my-1 text-base')
            ui.link('Lägg till ny spel', '/spel/add').classes(f'{css_prefix}-link ml-4 my-1 text-base')
            ui.link('Historik', '/spel/list').classes(f'{css_prefix}-link ml-4 my-1 text-base')
        ui.separator().classes('my-2')


def dashboard_content():
    with ui.column().classes('p-10 w-full max-w-xl'):
        ui.label('Dashboard').classes('text-3xl font-bold').style(f'color: {PRIMARY_COLOR}')
        ui.label('Välkommen till DeepPick AI! Här kan du hantera dina kuponger och spelomgångar.').classes('mb-4')
        ui.button('Lägg till ny kupong', on_click=lambda: ui.navigate.to('/kuponger/add')).classes('mr-4')
        ui.button('Lägg till ny spelomgång', on_click=lambda: ui.navigate.to('/spel/add')).classes('mr-4')



SEVERITY_OPTIONS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
STATUS_OPTIONS = ['NEW', 'IN_REVIEW', 'MITIGATED']

def spel_add_content():
    with ui.column().classes('p-10 w-full max-w-xl'):
        ui.label('Lägg till ny spelomgång').classes('text-3xl font-bold').style('color: #1F6C74')
        ui.label('Här kan du lägga till information om en ny spelomgång.').classes('mb-4')
        speltyp = ui.select(['Stryktipset', 'Europatipset'], label='Speltyp').classes('w-full')
        omgång = ui.input('Omgång (t.ex. 2024-09-15)').props('type=date').classes('w-full')
def kupong_add_content():
    with ui.column().classes('p-10 w-full max-w-xl'):
        ui.label('Lägg till ny kupong').classes('text-3xl font-bold').style('color: #1F6C74')
        ui.label('Här kan du lägga till information om en ny kupong.').classes('mb-4')
        speltyp = ui.select(['Stryktipset', 'Europatipset'], label='Speltyp').classes('w-full')
        speldag = ui.input('Speldag (YYYY-MM-DD)').props('type=date').classes('w-full')


def kupong_list_content():
    with ui.column().classes('p-10'):
        ui.label('Kuponger').classes('text-3xl font-bold').style(f'color: {PRIMARY_COLOR}')
        ui.label('Här kan du se alla kuponger.').classes('mb-4')
def spel_list_content():
    with ui.column().classes('p-10'):
        ui.label('Spelomgångar').classes('text-3xl font-bold').style(f'color: {PRIMARY_COLOR}')
        ui.label('Här kan du se alla spelomgångar.').classes('mb-4')


@ui.page('/')
@ui.page('/{path:path}')
def main_layout_page(path: str = ''):
    ui.add_css(f"""
    .{css_prefix}-header {{
        background: {PRIMARY_COLOR};
        color: white;
        box-shadow: 0 2px 12px #0002;
    }}
    .{css_prefix}-drawer {{
        background: {PRIMARY_COLOR};
        color: {TEXT_COLOR_LIGHT};
    }}
    .{css_prefix}-link {{
        color: {TEXT_COLOR_LIGHT};
        text-decoration: none;
        padding: 0.5rem 1rem;
        display: block;
        border-radius: 8px;
        font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }}
    .{css_prefix}-link:hover, .{css_prefix}-link--active {{
        color: {SECONDARY_COLOR};
        background: rgba(255,255,255,0.07);
        text-decoration: underline;
    }}
    """)
    print(f'main_layout_page anropad för sökväg: /{path}')

    left_drawer = ui.left_drawer()
    fill_main_header(left_drawer)
    fill_main_drawer(left_drawer)
    with ui.column().classes('w-full min-h-screen items-stretch'):
        ui.sub_pages(
            {
                '/': dashboard_content,
                '/kuponger/add': kupong_add_content,
                '/kuponger/list': kupong_list_content,
                '/spel/add': spel_add_content,
                '/spel/list': spel_list_content,
            }
        )
