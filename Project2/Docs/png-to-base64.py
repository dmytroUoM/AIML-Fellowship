import base64
from pathlib import Path

png_file = Path('scale.png')
out_file = Path('scale_base64.txt')

with open(png_file, 'rb') as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

out_file.write_text('data:image/png;base64,' + encoded, encoding='utf-8')

print(f'Saved Base64 output to: {out_file.resolve()}')