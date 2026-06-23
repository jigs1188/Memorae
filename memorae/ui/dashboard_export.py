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
    <title>Memorae — Personal Memory Intelligence Engine</title>
    <meta name="description" content="AI-powered personal memory query system. Analyzes WhatsApp, Slack, Gmail, Calendar and Notion to surface what matters most.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {{
            --bg:          #f5f0ff;
            --panel:       rgba(255, 255, 255, 0.9);
            --border:      rgba(139, 92, 246, 0.18);
            --purple-dark: #4c1d95;
            --purple:      #7c3aed;
            --purple-mid:  #8b5cf6;
            --purple-light:#a78bfa;
            --purple-pale: #ede9fe;
            --text:        #1e1033;
            --text-muted:  #6b7280;
            --green:       #059669;
            --amber:       #d97706;
            --red:         #dc2626;
            --gradient:    linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%);
            --shadow:      0 4px 24px rgba(124, 58, 237, 0.12);
            --shadow-lg:   0 8px 40px rgba(124, 58, 237, 0.18);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            overflow: hidden;
            background-image:
                radial-gradient(ellipse at 10% 60%, rgba(139, 92, 246, 0.1) 0%, transparent 55%),
                radial-gradient(ellipse at 90% 20%, rgba(76, 29, 149, 0.09) 0%, transparent 55%);
        }}

        /* ─── Sidebar ─────────────────────────────────────────────────────── */
        .sidebar {{
            width: 300px;
            min-width: 300px;
            background: var(--panel);
            backdrop-filter: blur(16px);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 0;
            overflow: hidden;
            box-shadow: 4px 0 20px rgba(124, 58, 237, 0.06);
        }}

        .sidebar-header {{
            padding: 1.75rem 1.5rem 1.25rem;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(237, 233, 254, 0.6) 0%, transparent 100%);
        }}

        .sidebar-header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 0.25rem;
        }}

        .sidebar-header .subtitle {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
            letter-spacing: 0.02em;
        }}

        .sidebar-queries {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem 1rem;
        }}

        .sidebar-label {{
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: var(--purple-mid);
            text-transform: uppercase;
            padding: 0 0.5rem;
            margin-bottom: 0.5rem;
        }}

        .query-item {{
            padding: 0.75rem 1rem;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 0.35rem;
            transition: all 0.18s ease;
            border: 1px solid transparent;
            line-height: 1.4;
        }}

        .query-item:hover {{
            background: var(--purple-pale);
            color: var(--purple-dark);
            border-color: var(--border);
            transform: translateX(2px);
        }}

        .query-item.active {{
            background: var(--gradient);
            color: white;
            border-color: transparent;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
        }}

        /* ─── Main content ────────────────────────────────────────────────── */
        .main-content {{
            flex: 1;
            padding: 2rem 2.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        /* ─── Cards ───────────────────────────────────────────────────────── */
        .card {{
            background: var(--panel);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: var(--shadow);
            animation: slideIn 0.3s ease both;
        }}

        .card:nth-child(2) {{ animation-delay: 0.07s; }}
        .card:nth-child(3) {{ animation-delay: 0.14s; }}
        .card:nth-child(4) {{ animation-delay: 0.21s; }}

        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .card-title {{
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--purple-mid);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card-title::before {{
            content: '';
            display: inline-block;
            width: 3px;
            height: 14px;
            background: var(--gradient);
            border-radius: 2px;
        }}

        /* ─── Query header card ───────────────────────────────────────────── */
        .query-header {{
            border-left: 3px solid var(--purple);
        }}

        .query-text {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--purple-dark);
            margin-bottom: 1rem;
            line-height: 1.4;
        }}

        .meta-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .badge-purple {{ background: var(--purple-pale); color: var(--purple-dark); }}
        .badge-green  {{ background: #d1fae5; color: #065f46; }}
        .badge-amber  {{ background: #fef3c7; color: #92400e; }}
        .badge-blue   {{ background: #dbeafe; color: #1e40af; }}

        /* ─── Answer card ─────────────────────────────────────────────────── */
        .answer-body {{
            font-size: 0.95rem;
            line-height: 1.75;
            color: var(--text);
        }}

        .answer-body p {{ margin-bottom: 0.75rem; }}
        .answer-body ul, .answer-body ol {{ padding-left: 1.5rem; margin-bottom: 0.75rem; }}
        .answer-body li {{ margin-bottom: 0.3rem; line-height: 1.6; }}
        .answer-body strong {{ color: var(--purple-dark); font-weight: 600; }}
        .answer-body h1, .answer-body h2, .answer-body h3 {{
            color: var(--purple-dark);
            margin-bottom: 0.5rem;
            margin-top: 1rem;
            font-size: 1rem;
        }}
        .answer-body pre, .answer-body code {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: inherit;
        }}
        .answer-body pre {{
            background: rgba(0,0,0,0.03);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }}

        /* ─── Contradiction alert ─────────────────────────────────────────── */
        .contradiction-box {{
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-left: 4px solid #f59e0b;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }}

        .contradiction-box .title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: #92400e;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .contradiction-box ul {{
            padding-left: 1.25rem;
            font-size: 0.875rem;
            color: #78350f;
            line-height: 1.6;
        }}

        /* ─── Reasoning section ───────────────────────────────────────────── */
        .reasoning-field {{
            margin-bottom: 1.1rem;
            padding-bottom: 1.1rem;
            border-bottom: 1px solid var(--border);
        }}

        .reasoning-field:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}

        .reasoning-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--purple);
            margin-bottom: 0.5rem;
        }}

        .reasoning-value {{
            font-size: 0.875rem;
            color: var(--text-muted);
            line-height: 1.65;
        }}

        .selected-event-item {{
            padding: 0.6rem 0.9rem;
            border-radius: 8px;
            background: var(--purple-pale);
            margin-bottom: 0.45rem;
            font-size: 0.8rem;
            line-height: 1.5;
        }}

        .selected-event-item .ev-source {{
            font-weight: 600;
            color: var(--purple-dark);
        }}

        .selected-event-item .ev-score {{
            float: right;
            background: var(--gradient);
            color: white;
            padding: 0.1rem 0.45rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        .selected-event-item .ev-reason {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        /* ─── Context events ──────────────────────────────────────────────── */
        .events-grid {{
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }}

        .event-card {{
            background: #faf5ff;
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }}

        .event-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .source-tag {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            background: var(--gradient);
            color: white;
        }}

        .score-tag {{
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--purple);
        }}

        .ts-tag {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-left: auto;
        }}

        .event-content {{
            font-size: 0.845rem;
            color: var(--text);
            line-height: 1.55;
        }}

        /* ─── Empty state ─────────────────────────────────────────────────── */
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            gap: 1rem;
            text-align: center;
        }}

        .empty-state svg {{ opacity: 0.4; }}
        .empty-state p {{ font-size: 0.95rem; }}

        /* ─── Scrollbar ───────────────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width: 5px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--purple-light); border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>Memorae</h1>
            <div class="subtitle">Personal Memory Intelligence Engine</div>
        </div>
        <div class="sidebar-queries">
            <div class="sidebar-label">Queries</div>
            <div id="query-list"></div>
        </div>
    </div>

    <div class="main-content" id="main-content">
        <div class="empty-state">
            <svg width="52" height="52" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
            </svg>
            <p>Select a query from the sidebar to view results</p>
        </div>
    </div>

    <script>
        const results = {json_data};

        // ── Sidebar ──────────────────────────────────────────────────────────
        function renderSidebar() {{
            const list = document.getElementById('query-list');
            results.forEach((res, i) => {{
                const el = document.createElement('div');
                el.className = 'query-item';
                el.textContent = res.query;
                el.onclick = () => selectQuery(i);
                list.appendChild(el);
            }});
        }}

        // ── Source color ─────────────────────────────────────────────────────
        const SOURCE_COLORS = {{
            calendar: '#7c3aed', gmail: '#dc2626', slack: '#1d4ed8',
            whatsapp: '#059669', notion: '#374151', sms: '#d97706',
            reminder: '#9333ea', chrome_extension: '#6b7280',
        }};
        function sourceStyle(src) {{
            const c = SOURCE_COLORS[src] || '#6b7280';
            return `background:${{c}};`;
        }}

        // ── Format reasoning ─────────────────────────────────────────────────
        function formatReasoningSection(key, value) {{
            const label = key.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());

            if (key === 'why_selected' && Array.isArray(value)) {{
                const items = value.map(v => {{
                    if (typeof v === 'object' && v !== null) {{
                        const score = (v.score || v.relevance_score || 0).toFixed(3);
                        const preview = v.content_preview || '';
                        const src = v.source || '';
                        const reason = v.reason || '';
                        const bm25 = v.score_breakdown && v.score_breakdown.bm25 !== undefined
                            ? ` · BM25: ${{v.score_breakdown.bm25.toFixed(3)}}` : '';
                        return `<div class="selected-event-item">
                            <span class="ev-score">${{score}}${{bm25}}</span>
                            <span class="ev-source">[${{src}}]</span> ${{preview}}
                            ${{reason ? `<div class="ev-reason">${{reason}}</div>` : ''}}
                        </div>`;
                    }}
                    return `<div class="selected-event-item">${{v}}</div>`;
                }}).join('');
                return `<div class="reasoning-field">
                    <div class="reasoning-label">${{label}} (${{value.length}} events)</div>
                    <div>${{items}}</div>
                </div>`;
            }}

            if (key === 'contradiction_resolution' && Array.isArray(value)) {{
                if (value.length === 0) return '';
                const items = value.map(v => `<li>${{v}}</li>`).join('');
                return `<div class="contradiction-box">
                    <div class="title">Contradiction / Update Resolution</div>
                    <ul>${{items}}</ul>
                </div>`;
            }}

            if (key === 'dropped_sample' && Array.isArray(value)) {{
                if (value.length === 0) return '';
                const items = value.map(v => {{
                    const preview = typeof v === 'object' ? (v.content_preview || JSON.stringify(v)) : v;
                    return `<li style="margin-bottom:0.3rem;font-size:0.8rem;">${{preview}}</li>`;
                }}).join('');
                return `<div class="reasoning-field">
                    <div class="reasoning-label">Dropped Sample (token overflow)</div>
                    <div class="reasoning-value"><ul style="padding-left:1.2rem">${{items}}</ul></div>
                </div>`;
            }}

            if (Array.isArray(value)) {{
                const text = value.join(', ');
                return `<div class="reasoning-field">
                    <div class="reasoning-label">${{label}}</div>
                    <div class="reasoning-value">${{text}}</div>
                </div>`;
            }}

            const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
            if (!text || text === '[]' || text === '{{}}') return '';
            return `<div class="reasoning-field">
                <div class="reasoning-label">${{label}}</div>
                <div class="reasoning-value">${{text}}</div>
            </div>`;
        }}

        // ── Main render ───────────────────────────────────────────────────────
        function selectQuery(index) {{
            document.querySelectorAll('.query-item').forEach((el, i) => {{
                el.classList.toggle('active', i === index);
            }});

            const res = results[index];
            const content = document.getElementById('main-content');
            const stats = res.context_stats || {{}};

            // --- Card 1: Query + metadata
            const modelBadge = res.model_used && res.model_used !== 'offline'
                ? `<span class="badge badge-purple">Model: ${{res.model_used}}</span>`
                : `<span class="badge badge-amber">Offline / No LLM</span>`;

            const card1 = `<div class="card query-header">
                <div class="card-title">Query</div>
                <div class="query-text">${{res.query}}</div>
                <div class="meta-row">
                    ${{modelBadge}}
                    ${{stats.token_estimate ? `<span class="badge badge-blue">~${{stats.token_estimate}} tokens</span>` : ''}}
                    ${{stats.events_used !== undefined ? `<span class="badge badge-green">${{stats.events_used}} events used</span>` : ''}}
                    ${{stats.events_dropped ? `<span class="badge badge-amber">${{stats.events_dropped}} dropped</span>` : ''}}
                </div>
            </div>`;

            // --- Contradiction notes (standalone alert box)
            let contradictionHtml = '';
            const notes = res.contradiction_notes || [];
            if (notes.length > 0) {{
                const items = notes.map(n => `<li>${{n}}</li>`).join('');
                contradictionHtml = `<div class="card">
                    <div class="contradiction-box" style="margin-bottom:0">
                        <div class="title">Contradiction / Update Resolution Detected</div>
                        <ul>${{items}}</ul>
                    </div>
                </div>`;
            }}

            // --- Card 2: Answer
            let answerHtml = '';
            if (res.answer && !res.answer.startsWith('[ERROR')) {{
                const parsed = typeof marked !== 'undefined' ? marked.parse(res.answer) : res.answer.replace(/\\n/g, '<br>');
                answerHtml = `<div class="card">
                    <div class="card-title">AI Answer</div>
                    <div class="answer-body">${{parsed}}</div>
                </div>`;
            }} else if (res.answer) {{
                answerHtml = `<div class="card">
                    <div class="card-title">Offline Mode — Context Preview</div>
                    <div class="answer-body" style="color:var(--text-muted);font-style:italic;">
                        Run <code>python main.py</code> with an LLM API key to generate answers.<br>
                        In offline mode, the selected context events are shown below.
                    </div>
                </div>`;
            }}

            // --- Card 3: Reasoning
            let reasoningBody = '';
            if (res.reasoning && typeof res.reasoning === 'object') {{
                const SKIP = ['contradiction_resolution']; // handled separately
                for (const [k, v] of Object.entries(res.reasoning)) {{
                    if (SKIP.includes(k)) continue;
                    reasoningBody += formatReasoningSection(k, v);
                }}
            }}
            const reasoningHtml = reasoningBody ? `<div class="card">
                <div class="card-title">Engine Reasoning</div>
                ${{reasoningBody}}
            </div>` : '';

            // --- Card 4: Selected Context Events
            const eventsHtml = (res.selected_context || []).map(ctx => `
                <div class="event-card">
                    <div class="event-card-header">
                        <span class="source-tag" style="${{sourceStyle(ctx.source)}}">${{ctx.source}}</span>
                        <span class="score-tag">score: ${{ctx.relevance_score !== undefined ? ctx.relevance_score.toFixed(3) : 'N/A'}}</span>
                        <span class="ts-tag">${{ctx.timestamp ? ctx.timestamp.slice(0, 16).replace('T', ' ') + ' UTC' : ''}}</span>
                    </div>
                    <div class="event-content">${{ctx.content || ''}}</div>
                </div>
            `).join('');

            const contextHtml = eventsHtml ? `<div class="card">
                <div class="card-title">Selected Context (${{(res.selected_context || []).length}} events)</div>
                <div class="events-grid">${{eventsHtml}}</div>
            </div>` : '';

            content.innerHTML = card1 + contradictionHtml + answerHtml + reasoningHtml + contextHtml;
        }}

        // ── Init ─────────────────────────────────────────────────────────────
        renderSidebar();
        if (results.length > 0) selectQuery(0);
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
