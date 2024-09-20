import cairosvg
from PIL import Image
import io


SQUARE_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <path d="M50,50 L250,50 L250,250 L50,250 Z" fill="none" stroke="#{COLOR}" stroke-width="4" />
  <text x="150" y="140" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#{COLOR}" text-anchor="middle">{UPPER_TEXT}</text>
  <text x="150" y="210" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#{COLOR}" text-anchor="middle">{LOWER_TEXT}</text>
</svg>
"""
# font-style="italic" 

def square_logo(upper_text, lower_text, color='000000'):
    svg_code = SQUARE_LOGO_SVG.format(UPPER_TEXT=upper_text, LOWER_TEXT=lower_text, COLOR=color)
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data))