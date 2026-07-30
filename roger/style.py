"""The Roger app's visual system — v2, from the Claude Design project
"Roger: Codebase quiz app" (Roger v2.dc.html): ivory paper, ink text, one
clay accent, serif display from system stacks, and a warm-ink code panel.

Adaptations from the delivered CSS, all mechanical:
- Our container keys carry suffixes, so exact st-key classes become
  substring selectors (the design already wrote them that way).
- The design welds a .rog-file caption bar to the code block with a
  sibling selector; Streamlit wraps each element in its own container,
  so the weld is reproduced through a keyed wrapper (st-key-codepanel).

Everything is local: system font stacks only, no URLs of any kind.
"""

STYLE = """
<style>
:root{
  --paper:#F7F4EF; --card:#FBF8F3; --sunken:#EFE9DE; --track:#EBE5DA;
  --line:#E4DDD1; --line-soft:#EDE7DC; --ink:#191713; --ink-2:#4A443C;
  --muted:#8A8175; --faint:#A39A8C;
  --clay:#C1683F; --clay-press:#B05A34; --ok:#4C7A4A; --no:#A32C22;
  --code-bg:#1F1C18; --code-gutter:#1B1814; --code-fg:#E8E1D5;
  --serif:'Iowan Old Style','Palatino Nova',Palatino,Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}

/* ---------- page ---------- */
.stApp{ background:var(--paper); }
[data-testid="stMainBlockContainer"]{
  max-width:820px; padding:1.6rem 3.2rem 7rem; margin:0 auto;
}
html,body,[class*="css"]{ font-family:var(--sans); color:var(--ink);
  -webkit-font-smoothing:antialiased; }
[data-testid="stHeader"]{ background:transparent; height:0; visibility:hidden; }
#MainMenu, footer{ visibility:hidden; height:0; }
h1,h2,h3,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3{
  font-family:var(--serif) !important; font-weight:400 !important;
  letter-spacing:-.016em; }
h1, [data-testid="stMarkdownContainer"] h1{ font-size:2.5rem; line-height:1.14; }

/* ---------- app header (st.markdown HTML, one row) ---------- */
.rog-head{ display:flex; align-items:center; justify-content:space-between;
  padding:14px 0 16px; border-bottom:1px solid var(--line); margin-bottom:18px; }
.rog-repo{ display:flex; align-items:center; gap:9px; font-family:var(--mono);
  font-size:12.5px; color:var(--ink-2); }
.rog-repo:before{ content:""; width:7px; height:7px; border-radius:50%;
  background:var(--clay); }
.rog-badge{ display:inline-flex; align-items:center; gap:8px;
  padding:5px 11px 5px 9px; border:1px solid #E0D8CA; border-radius:999px;
  background:#F4F0E8; font-family:var(--mono); font-size:11.5px;
  color:var(--ink-2); }
.rog-badge b{ font-weight:400; color:var(--muted); }
.rog-badge:before{ content:""; width:6px; height:6px; border-radius:50%;
  background:var(--ok); box-shadow:0 0 0 3px rgba(76,122,74,.14); }
.rog-badge.busy:before{ background:var(--clay);
  box-shadow:0 0 0 3px rgba(193,104,63,.16); animation:rogPulse 1.4s ease-in-out infinite; }
@keyframes rogPulse{ 0%,100%{opacity:.42} 50%{opacity:1} }

/* ---------- top navigation + count picker (stButtonGroup) ---------- */
[data-testid="stButtonGroup"]{ gap:3px; padding:3px; border-radius:12px;
  background:var(--track); width:fit-content; }
[data-testid="stButtonGroup"] button{ border:none; border-radius:9px;
  background:transparent; padding:8px 22px; font-family:var(--sans);
  font-size:13.5px; color:#7C756A; letter-spacing:-.006em;
  transition:background .16s ease, color .16s ease; }
[data-testid="stButtonGroup"] button:hover{ color:var(--ink); }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"]{
  background:var(--card); color:var(--ink); font-weight:600;
  box-shadow:0 1px 2px rgba(60,44,24,.09); }

/* ---------- answer option rows (div.stButton > button) ---------- */
div.stButton > button{
  all:unset; box-sizing:border-box; display:flex; align-items:center; gap:13px;
  width:100%; padding:12px 16px; margin:0 0 9px;
  border:1px solid var(--line); border-radius:11px; background:var(--card);
  font:400 14.5px/1.45 var(--sans); letter-spacing:-.006em; color:var(--ink);
  text-align:left; cursor:pointer;
  transition:border-color .16s ease, background .16s ease,
             transform .16s ease, box-shadow .16s ease;
}
div.stButton > button:hover{ border-color:var(--clay); transform:translateY(-1px);
  box-shadow:0 2px 6px rgba(120,58,26,.08); }
div.stButton > button:active{ transform:translateY(0); background:var(--sunken); }
div.stButton > button:focus-visible{ outline:2px solid var(--clay);
  outline-offset:2px; }
/* the A–D chip */
div.stButton > button:before{
  flex:none; width:23px; height:23px; display:flex; align-items:center;
  justify-content:center; border-radius:7px; background:var(--sunken);
  color:#6B655C; font:700 11.5px var(--mono);
  transition:background .16s ease, color .16s ease;
}
[class*="st-key-optA"] div.stButton > button:before{ content:"A"; }
[class*="st-key-optB"] div.stButton > button:before{ content:"B"; }
[class*="st-key-optC"] div.stButton > button:before{ content:"C"; }
[class*="st-key-optD"] div.stButton > button:before{ content:"D"; }

/* graded states — key fragments: stateok / stateno / stateans / statemut */
[class*="stateok"] div.stButton > button{ border-color:var(--ok);
  background:rgba(76,122,74,.07); pointer-events:none;
  animation:rogReveal .22s ease-out; }
[class*="stateok"] div.stButton > button:before{ background:var(--ok);
  color:#FDFBF7; }
[class*="stateno"] div.stButton > button{ border-color:var(--no);
  background:rgba(163,44,34,.055); pointer-events:none;
  animation:rogReveal .22s ease-out; }
[class*="stateno"] div.stButton > button:before{ background:var(--no);
  color:#FDFBF7; }
[class*="stateans"] div.stButton > button{ border-color:var(--ok);
  background:var(--card); pointer-events:none; }
[class*="stateans"] div.stButton > button:before{ background:rgba(76,122,74,.14);
  color:#3C6339; }
[class*="statemut"] div.stButton > button{ border-color:var(--line-soft);
  background:var(--paper); color:var(--faint); pointer-events:none; }
[class*="statemut"] div.stButton > button:before{ background:#F1EBE0;
  color:#B5AB9C; }
@keyframes rogReveal{ from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:none} }

/* primary / dark / quiet buttons */
[class*="st-key-primary"] div.stButton > button{ width:auto;
  justify-content:center; padding:12px 30px; border:none; border-radius:11px;
  background:var(--clay); color:#FDFBF7; font-weight:600;
  box-shadow:0 1px 2px rgba(120,58,26,.28), inset 0 1px 0 rgba(255,255,255,.16); }
[class*="st-key-primary"] div.stButton > button:before{ content:none; }
[class*="st-key-primary"] div.stButton > button:hover{
  background:var(--clay-press); transform:translateY(-1px); }
[class*="st-key-next"] div.stButton > button{ width:auto;
  justify-content:center; padding:9px 22px; border:none; border-radius:10px;
  background:#211E19; color:var(--card); font-size:13.5px; font-weight:600;
  box-shadow:0 1px 2px rgba(33,30,25,.22), inset 0 1px 0 rgba(255,255,255,.1); }
[class*="st-key-next"] div.stButton > button:before{ content:none; }
[class*="st-key-next"] div.stButton > button:hover{ background:#332E27; }
[class*="st-key-quiet"] div.stButton > button{ width:auto;
  justify-content:center; padding:10px 24px; border:1px solid #DDD4C4;
  border-radius:10px; font-size:13.5px; font-weight:600; }
[class*="st-key-quiet"] div.stButton > button:before{ content:none; }
[class*="st-key-link"] div.stButton > button{ width:auto; padding:6px 0;
  border:none; background:none; color:#B45A33; font-size:13.5px; }
[class*="st-key-link"] div.stButton > button:before{ content:none; }
[class*="st-key-link"] div.stButton > button:hover{ color:#8F4525;
  text-decoration:underline; transform:none; box-shadow:none; }

/* ---------- question caption + explanation ---------- */
.rog-caption{ display:flex; align-items:center; gap:9px; font-size:11.5px;
  color:var(--muted); }
.rog-caption b{ color:var(--ink-2); font-variant-numeric:tabular-nums;
  letter-spacing:.04em; }
.rog-caption code{ font-family:var(--mono); background:none; border:none;
  padding:0; color:var(--ink-2); }
.rog-q{ font-family:var(--serif); font-size:25px; line-height:1.34;
  letter-spacing:-.012em; margin:14px 0 24px; text-wrap:pretty; }
.rog-why{ display:flex; gap:14px; margin:22px 0 0;
  animation:rogReveal .28s ease-out; }
.rog-why:before{ content:""; flex:none; width:2px; background:#DED6C7;
  border-radius:1px; }
.rog-why em{ font-family:var(--serif); font-size:16px; line-height:1.55;
  color:#5C564D; }

/* ---------- code (st.code, line_numbers=True) ---------- */
[data-testid="stCode"], [data-testid="stCodeBlock"]{
  border-radius:13px; overflow:hidden; background:var(--code-bg);
  box-shadow:0 1px 2px rgba(31,28,24,.2), 0 12px 30px rgba(31,28,24,.14);
  margin:0 0 6px;
}
[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre{
  background:var(--code-bg); border:none; padding:16px 22px 16px 0; margin:0;
}
[data-testid="stCode"] code, [data-testid="stCodeBlock"] code{
  font-family:var(--mono); font-size:13.5px; line-height:1.75;
  letter-spacing:-.002em; color:var(--code-fg);
}
[data-testid="stCode"] .linenos, [data-testid="stCode"] span[data-line-number],
[data-testid="stCodeBlock"] .linenos{
  display:inline-block; width:52px; padding-right:16px; margin-right:20px;
  text-align:right; color:#5E574C; background:var(--code-gutter);
  user-select:none;
}
[data-testid="stCode"] button, [data-testid="stCodeBlock"] button{
  opacity:0; color:#9A9287; transition:opacity .16s ease; }
[data-testid="stCode"]:hover button, [data-testid="stCodeBlock"]:hover button{
  opacity:1; }
/* pygments tokens on the warm-ink surface */
[data-testid="stCode"] .k, [data-testid="stCode"] .kn, [data-testid="stCode"] .kd,
[data-testid="stCode"] .ow{ color:#D98A73; }
[data-testid="stCode"] .s, [data-testid="stCode"] .s1, [data-testid="stCode"] .s2,
[data-testid="stCode"] .sd{ color:#B4C99A; }
[data-testid="stCode"] .c, [data-testid="stCode"] .c1,
[data-testid="stCode"] .cm{ color:#7E776B; font-style:normal; }
[data-testid="stCode"] .m, [data-testid="stCode"] .mi,
[data-testid="stCode"] .mf{ color:#C9A96A; }
[data-testid="stCode"] .nf, [data-testid="stCode"] .fm{ color:#8FB8D0; }
[data-testid="stCode"] .nc, [data-testid="stCode"] .nb,
[data-testid="stCode"] .nd{ color:#C7A8D8; }
/* file caption sitting above a block */
.rog-file{ display:flex; align-items:center; justify-content:space-between;
  padding:10px 14px 10px 18px; border-radius:13px 13px 0 0;
  background:var(--code-bg); border-bottom:1px solid #2E2A24;
  font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.rog-file + [data-testid="stCode"]{ border-radius:0 0 13px 13px; }

/* ---------- markdown doc excerpts: card + tables ---------- */
[class*="st-key-docpanel"] [data-testid="stMarkdown"]{
  background:var(--card); border:1px solid var(--line); border-radius:13px;
  padding:16px 20px 10px; }
.stMarkdown table{ border-collapse:collapse; width:100%; margin:4px 0 12px;
  font-size:13.5px; }
.stMarkdown thead tr{ border-bottom:1px solid var(--line); }
.stMarkdown th{ font-weight:600; text-align:left; padding:7px 16px 9px 0;
  color:var(--ink-2); font-size:11.5px; letter-spacing:.05em;
  text-transform:uppercase; border:none; }
.stMarkdown td{ padding:8px 16px 8px 0; border:none;
  border-bottom:1px solid var(--line-soft); vertical-align:top;
  line-height:1.5; }
.stMarkdown tbody tr:last-child td{ border-bottom:none; }

/* inline code in markdown */
.stMarkdown code{ font-family:var(--mono); font-size:.86em;
  background:var(--sunken); border:1px solid #E2DACC; border-radius:6px;
  padding:1px 6px; color:var(--ink); }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{ background:transparent; padding:0;
  margin:0 0 28px; gap:0; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"]{ display:none; }
[class*="st-key-turnuser"] [data-testid="stChatMessage"]{ display:flex;
  justify-content:flex-end; }
[class*="st-key-turnuser"] [data-testid="stChatMessageContent"]{
  max-width:74%; padding:11px 16px; border:1px solid var(--line);
  border-radius:15px 15px 5px 15px; background:var(--sunken);
  font-size:14.5px; line-height:1.5; letter-spacing:-.006em; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"]{
  font-size:15px; line-height:1.7; letter-spacing:-.004em; color:#241F19;
  text-wrap:pretty; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"]:before{
  content:"● ROGER"; display:block; margin-bottom:13px;
  font:700 11px var(--sans); letter-spacing:.14em; color:var(--muted); }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] p{ margin:0 0 12px; }
/* long "complete list" answers stay scannable */
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] ul{
  list-style:none; margin:6px 0 12px; padding:0;
  counter-reset:rog; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] li{
  counter-increment:rog; position:relative; display:block;
  padding:7px 0 7px 32px; margin:0; border-bottom:1px solid var(--line-soft);
  font-size:14px; line-height:1.5; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] li:before{
  content:counter(rog); position:absolute; left:0; top:9px;
  font-family:var(--mono); font-size:11px;
  color:#B5AB9C; font-variant-numeric:tabular-nums; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] li code{
  background:none; border:none; padding:0; font-size:13px; }
.rog-thinking{ font-family:var(--serif); font-size:17px; color:#5C564D;
  animation:rogPulse 1.7s ease-in-out infinite; }
.rog-scan{ font-family:var(--mono); font-size:11.5px; color:var(--faint); }

/* ---------- sources expander ---------- */
[data-testid="stExpander"]{ border:none; border-top:1px solid var(--line);
  border-radius:0; background:transparent; margin-top:8px; }
[data-testid="stExpander"] summary{ padding:14px 0 0; font-size:12.5px;
  font-weight:600; color:#5C564D; transition:color .16s ease; }
[data-testid="stExpander"] summary:hover{ color:var(--ink); }
[data-testid="stExpander"] summary svg{ fill:var(--faint); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"]{
  padding:12px 0 4px 17px; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p{
  font-family:var(--mono); font-size:12.5px; line-height:2; color:var(--ink-2);
  margin:0; }

/* ---------- progress ---------- */
[data-testid="stProgress"] > div > div{ height:5px; border-radius:3px;
  background:#E6DFD2; }
[data-testid="stProgress"] > div > div > div{ background:var(--clay);
  border-radius:3px; transition:width .2s ease; }
[data-testid="stProgress"] p{ font-family:var(--serif); font-size:27px;
  color:var(--ink); letter-spacing:-.012em; text-align:center; margin:0 0 14px; }

/* ---------- chat input (pinned: lives outside the nav switch) ---------- */
[data-testid="stBottomBlockContainer"]{ background:var(--card);
  border-top:1px solid var(--line-soft); padding:16px 3.2rem 26px;
  max-width:820px; margin:0 auto; }
[data-testid="stChatInput"]{ border:1px solid #DDD4C4; border-radius:12px;
  background:var(--paper); padding:3px 5px;
  transition:border-color .16s ease, box-shadow .16s ease; }
[data-testid="stChatInput"]:focus-within{ border-color:var(--clay);
  box-shadow:0 0 0 3px rgba(193,104,63,.1); }
[data-testid="stChatInput"] textarea{ font-family:var(--sans);
  font-size:14.5px; letter-spacing:-.005em; color:var(--ink); }
[data-testid="stChatInput"] textarea::placeholder{ color:var(--faint); }
[data-testid="stChatInput"] button{ border-radius:8px; background:var(--clay);
  color:#FDFBF7; transition:background .16s ease; }
[data-testid="stChatInput"] button:hover{ background:var(--clay-press); }
[data-testid="stChatInput"][data-disabled="true"],
[data-testid="stChatInput"]:has(textarea:disabled){ background:#F1EBE0;
  border-color:#E6DFD2; }
.rog-privacy{ font-size:11.5px; color:var(--faint); text-align:center;
  margin:10px 0 0; }

/* ---------- weld: file bar + code block inside one keyed panel ---------- */
[class*="st-key-codepanel"] div[data-testid="stVerticalBlock"]{ gap:0; }
[class*="st-key-codepanel"] [data-testid="stElementContainer"]{ margin:0; }
[class*="st-key-codepanel"] .rog-file{ margin-bottom:0; }
[class*="st-key-codepanel"] [data-testid="stCode"],
[class*="st-key-codepanel"] [data-testid="stCodeBlock"]{
  border-radius:0 0 13px 13px; }
</style>
"""

# Pinned-light theme (the code panel is already dark; a dark app would
# flatten the one contrast the design leans on). Applied per-process by
# the launcher — never written to the user's ~/.streamlit.
THEME_ENV = {
    "STREAMLIT_THEME_BASE": "light",
    "STREAMLIT_THEME_PRIMARY_COLOR": "#C1683F",
    "STREAMLIT_THEME_BACKGROUND_COLOR": "#F7F4EF",
    "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": "#EFE9DE",
    "STREAMLIT_THEME_TEXT_COLOR": "#191713",
    "STREAMLIT_THEME_FONT": "sans-serif",
    "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
}
