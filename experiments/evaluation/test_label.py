import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 查看所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(c in f for c in ['宋体', '黑体', '微软', 'Sim', 'Song', 'Hei'])]

print("可用的中文字体:", chinese_fonts)