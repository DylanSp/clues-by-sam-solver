from playwright.sync_api import sync_playwright


def complete_puzzle(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # demo of browser automation
        page.screenshot(path="example.png")

        browser.close()
