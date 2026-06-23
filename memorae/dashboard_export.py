import json
from pathlib import Path

def generate_dashboard(results_dict: list[dict], output_path: str | Path) -> None:
    """Generates a beautiful, self-contained HTML dashboard with embedded results."""
    
    json_data = json.dumps(results_dict, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memorae Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {{
            --bg-dark: #fcfaff;
            --bg-panel: rgba(255, 255, 255, 0.85);
            --border-color: rgba(147, 51, 234, 0.15);
            --text-main: #3b0764;
            --text-muted: #6b7280;
            --accent: #9333ea;
            --accent-glow: rgba(147, 51, 234, 0.3);
            --gradient: linear-gradient(135deg, #a855f7, #7e22ce);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            overflow: hidden;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(168, 85, 247, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(126, 34, 206, 0.08) 0%, transparent 50%);
        }}

        /* Layout */
        .sidebar {{
            width: 320px;
            background: var(--bg-panel);
            backdrop-filter: blur(12px);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            z-index: 10;
        }}

        .main-content {{
            flex: 1;
            padding: 2.5rem;
            overflow-y: auto;
            position: relative;
        }}

        /* Typography */
        h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
            letter-spacing: -0.5px;
        }}

        h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #4c1d95;
        }}

        h3 {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            margin-top: 2rem;
        }}

        /* Query List */
        .query-item {{
            padding: 1rem;
            border-radius: 12px;
            cursor: pointer;
            margin-bottom: 0.75rem;
            background: rgba(147, 51, 234, 0.03);
            border: 1px solid transparent;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 0.95rem;
            line-height: 1.4;
        }}

        .query-item:hover {{
            background: rgba(147, 51, 234, 0.08);
            transform: translateY(-1px);
        }}

        .query-item.active {{
            background: rgba(147, 51, 234, 0.12);
            border-color: rgba(147, 51, 234, 0.3);
            box-shadow: 0 0 15px var(--accent-glow);
        }}

        /* Cards & Content */
        .glass-card {{
            background: var(--bg-panel);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.4s ease-out;
        }}

        .answer-text {{
            font-size: 1.1rem;
            line-height: 1.7;
            color: #374151;
            /* Markdown styles */
        }}
        .answer-text h2, .answer-text h3 {{ margin-top: 1.5em; margin-bottom: 0.5em; color: #4c1d95; }}
        .answer-text p {{ margin-bottom: 1em; }}
        .answer-text ul, .answer-text ol {{ margin-left: 1.5em; margin-bottom: 1em; }}
        .answer-text li {{ margin-bottom: 0.25em; }}
        .answer-text strong {{ color: #3b0764; }}

        /* Metadata Tags */
        .meta-container {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}

        .tag {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            background: rgba(147, 51, 234, 0.08);
            border: 1px solid rgba(147, 51, 234, 0.2);
            color: #7e22ce;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .tag.model {{ background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.2); }}
        .tag.tokens {{ background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.2); }}

        /* Source Events */
        .event-card {{
            background: rgba(243, 232, 255, 0.6);
            border-left: 3px solid var(--accent);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
            color: #4b5563;
        }}

        .event-header {{
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }}

        /* Reasoning */
        .reasoning-text {{
            font-size: 0.95rem;
            color: var(--text-muted);
            line-height: 1.6;
            white-space: pre-wrap;
        }}

        /* Empty State */
        .empty-state {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            flex-direction: column;
            gap: 1rem;
            opacity: 0.7;
        }}

        /* Animations */
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>

    <div class="sidebar">
        <h1>Memorae</h1>
        <div id="query-list"></div>
    </div>

    <div class="main-content" id="main-content">
        <div class="empty-state">
            <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
            <p>Select a query to view results</p>
        </div>
    </div>

    <script>
        const results = {json_data};

        function renderSidebar() {{
            const list = document.getElementById('query-list');
            results.forEach((res, index) => {{
                const el = document.createElement('div');
                el.className = 'query-item';
                el.innerText = res.query;
                el.onclick = () => selectQuery(index);
                list.appendChild(el);
            }});
        }}

        function selectQuery(index) {{
            // Update active state
            document.querySelectorAll('.query-item').forEach((el, i) => {{
                el.classList.toggle('active', i === index);
            }});

            const res = results[index];
            const content = document.getElementById('main-content');
            
            // Format events
            const eventsHtml = res.selected_context.map(ctx => `
                <div class="event-card">
                    <div class="event-header">
                        <span>Source: ${{ctx.source}}</span>
                        <span>Score: ${{ctx.relevance_score ? ctx.relevance_score.toFixed(3) : "N/A"}}</span>
                    </div>
                    <div>${{ctx.content}}</div>
                </div>
            `).join('');

            // Format reasoning
            let reasoningHtml = res.reasoning;
            if (typeof res.reasoning === 'object') {{
                let items = [];
                for (const [key, value] of Object.entries(res.reasoning)) {{
                    let formattedValue = value;
                    if (Array.isArray(value)) {{
                        formattedValue = value.map(v => {{
                            if (typeof v === 'object' && v !== null) {{
                                if (v.content_preview) {{
                                    return `<li style="margin-bottom: 0.5rem"><strong>[${{v.source}}]</strong> ${{v.content_preview}} <br><em style="font-size: 0.85rem; color: #6b7280;">Score: ${{v.score || v.relevance_score}} - ${{v.reason}}</em></li>`;
                                }}
                                return `<li>${{JSON.stringify(v)}}</li>`;
                            }}
                            return `<li>${{v}}</li>`;
                        }}).join('');
                        formattedValue = `<ul style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">${{formattedValue}}</ul>`;
                    } else if (typeof value === 'object' && value !== null) {{
                        formattedValue = JSON.stringify(value);
                    }}
                    items.push(`<div style="margin-bottom: 0.75rem;"><strong style="color: #4c1d95;">${{key.replace(/_/g, ' ').toUpperCase()}}:</strong> ${{formattedValue}}</div>`);
                }}
                reasoningHtml = `<div style="font-size: 0.95rem; color: #4b5563;">${{items.join('')}}</div>`;
            }}

            content.innerHTML = `
                <div class="glass-card">
                    <div class="meta-container">
                        <div class="tag model">AI Model: ${{res.model_used || 'Local/Offline'}}</div>
                        <div class="tag tokens">~${{res.context_stats.token_estimate}} tokens analyzed</div>
                        <div class="tag">${{res.context_stats.events_used}} events extracted</div>
                    </div>
                    <h2>${{res.query}}</h2>
                    <div class="answer-text">${{marked.parse(res.answer)}}</div>
                </div>

                <div class="glass-card" style="animation-delay: 0.1s">
                    <h3>Engine Reasoning</h3>
                    <div class="reasoning-text">${{reasoningHtml}}</div>
                </div>

                <div class="glass-card" style="animation-delay: 0.2s">
                    <h3>Selected Context (${{res.selected_context.length}} events)</h3>
                    ${{eventsHtml}}
                </div>
            `;
        }}

        // Initialize
        renderSidebar();
        if (results.length > 0) {{
            selectQuery(0);
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
