"""
EthicaAI Paper Converter V2 (Academic Style Polish)
Markdown 논문을 학술지 스타일(2단 컬럼)의 HTML로 변환합니다.
"""
import markdown
import os
import re

def convert_md_to_html(md_path, html_path, css_style="academic"):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Markdown 변환
    # extra: footnotes, tables, attr_list, def_list, abbr
    html_content = markdown.markdown(text, extensions=['extra', 'toc', 'fenced_code'])
    
    # 이미지 경로 수정
    def replace_img_path(match):
        alt = match.group(1)
        path = match.group(2)
        filename = os.path.basename(path)
        # 캡션 스타일링: 이미지 아래에 캡션 추가
        return f'<figure><img src="figures/{filename}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'
    
    html_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img_path, html_content)
    
    # *Fig X. ...* 형태의 텍스트를 캡션 스타일로 변환 (이미지 바로 뒤에 오는 기울임꼴)
    html_content = re.sub(r'<p><em>(Fig \d+\..*?)<\/em><\/p>', r'<div class="caption-text">\1</div>', html_content)

    html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>EthicaAI Final Paper</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Noto+Serif+KR:wght@300;400;600;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Noto Serif KR', 'Libre Baskerville', serif;
            line-height: 1.6;
            color: #222;
            margin: 0;
            padding: 0;
            background-color: #f0f2f5;
        }}
        
        .page {{
            max-width: 210mm;
            min-height: 297mm;
            margin: 20px auto;
            background-color: white;
            padding: 20mm;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            position: relative;
        }}
        
        h1 {{ 
            font-family: 'Roboto', sans-serif;
            font-size: 24pt; 
            font-weight: 700;
            text-align: center; 
            margin-bottom: 5mm;
            letter-spacing: -0.5px;
            color: #111;
        }}
        
        h2 {{ 
            font-family: 'Roboto', sans-serif;
            font-size: 14pt; 
            border-bottom: 2px solid #222;
            padding-bottom: 3px;
            margin-top: 20px;
            color: #333;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        h3 {{
            font-family: 'Roboto', sans-serif;
            font-size: 11pt;
            font-weight: 700;
            color: #444;
            margin-top: 15px;
            margin-bottom: 5px;
        }}
        
        .abstract-container {{
            background-color: #f8f9fa;
            border-left: 3px solid #222;
            padding: 15px 20px;
            margin: 20px 0 30px 0;
            font-size: 0.9em;
            text-align: justify;
        }}
        
        .abstract-title {{
            font-weight: bold;
            font-family: 'Roboto', sans-serif;
            text-transform: uppercase;
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
            display: block;
        }}

        /* 2단 컬럼 레이아웃 (본문) */
        .content-body {{
            column-count: 2;
            column-gap: 8mm;
            text-align: justify;
            font-size: 9.5pt;
        }}
        
        /* 1단 적용 요소 (제목, 초록, 큰 그림) */
        h1, .abstract-container, table, pre, .full-width {{
            column-span: all;
        }}
        
        figure {{
            margin: 15px 0;
            text-align: center;
            background-color: #fff;
            break-inside: avoid; /* 컬럼 중간에서 잘리지 않게 */
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #eee;
        }}
        
        figcaption {{
            font-family: 'Roboto', sans-serif;
            font-size: 8pt;
            color: #666;
            margin-top: 5px;
            font-weight: 500;
        }}
        
        .caption-text {{
            font-family: 'Roboto', sans-serif;
            font-size: 8.5pt;
            color: #444;
            text-align: center;
            font-style: italic;
            margin-bottom: 15px;
        }}

        blockquote {{
            border-left: 2px solid #3498db;
            margin: 10px 0;
            padding-left: 10px;
            color: #555;
            font-size: 0.9em;
            font-style: italic;
        }}
        
        code {{
            font-family: 'Consolas', monospace;
            background-color: #f4f4f4;
            padding: 1px 3px;
            border-radius: 2px;
            font-size: 0.85em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
            margin: 15px 0;
            break-inside: avoid;
        }}
        
        th, td {{
            border-top: 1px solid #ddd;
            border-bottom: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }}
        th {{ border-top: 2px solid #222; border-bottom: 2px solid #222; font-weight: 600; }}
        

        /* Print Style */
        @media print {{
            body {{ background: none; }}
            .page {{ 
                margin: 0; 
                box-shadow: none; 
                width: 100%;
                max-width: none;
                padding: 0;
            }}
            .content-body {{ column-count: 2; }}
        }}
    </style>
</head>
<body>
    <div class="page">
        {html_content}
    </div>
</body>
</html>
    """
    
    # Wrap abstract in container
    # (Regex to find ## Abstract ... content ... ##) - simple replace for now
    html_content = html_content.replace('<h2>Abstract (Korean)</h2>', '<div class="abstract-container"><span class="abstract-title">Abstract (Korean)</span>')
    html_content = html_content.replace('<h2>Abstract (English)</h2>', '</div><div class="abstract-container"><span class="abstract-title">Abstract (English)</span>')
    # Close second abstract div manually or rely on luck? Better to wrap entire content.
    # Hack: Inject closing div before first H2 that is NOT abstract
    
    # Let's apply specific class to the body content wrapper
    html_template = html_template.replace('{html_content}', f'<div class="content-body">{html_content}</div>')

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated Academic Style HTML: {html_path}")

if __name__ == "__main__":
    convert_md_to_html("paper_draft.md", "submission/paper.html")
