"""The Roger app's visual system — from the Claude Design project
"Roger: Codebase quiz app" (Roger.dc.html), adapted to Streamlit's DOM.

Two adaptations from the delivered CSS, both mechanical:
- Keyed containers: st.container(key="x") renders class st-key-x, but our
  keys carry per-question suffixes, so selectors match on substrings
  ([class*="st-key-optA"]) instead of exact classes.
- A–D chips come from :before content per letter class — no HTML inside
  buttons, exactly as the design's build notes prescribe.

Everything is local: system font stacks only, no URLs of any kind.
"""

STYLE = """
<style>
/* ---------- shell ---------- */
.stApp { background:#FFFFFF; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
  max-width:760px !important; margin:0 auto !important;
  padding-top:2.2rem; padding-bottom:6rem;
}
html, body, [class*="css"] { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
  Roboto,sans-serif; -webkit-font-smoothing:antialiased; color:#1D1D1F; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }

.roger-brand { font-size:13px; color:#6E6E73; letter-spacing:-.005em; margin-bottom:4px;
  display:flex; justify-content:space-between; align-items:baseline; }
.roger-brand b { color:#1D1D1F; font-weight:600; }
.roger-brand .backend { font-size:11.5px; color:#A1A1A6;
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
.roger-sub code { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:.9em; background:#F2F2EF; border-radius:4px; padding:0 4px; }
.roger-title { font-size:26px; font-weight:600; letter-spacing:-.02em; margin:6px 0 2px; }
.roger-sub { font-size:14.5px; line-height:1.6; color:#6E6E73; letter-spacing:-.005em; }
.roger-qcaption { font-size:12px; font-weight:600; letter-spacing:.07em;
  text-transform:uppercase; color:#86868B; margin-bottom:2px; }
.roger-qcaption .path { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  text-transform:none; letter-spacing:0; font-weight:400; color:#A1A1A6; }
.roger-codecaption { display:flex; justify-content:space-between; font-size:12px;
  color:#A1A1A6; margin:14px 0 4px; }
.roger-codecaption .path { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
.roger-score { font-size:34px; font-weight:600; letter-spacing:-.02em; }
.roger-score small { font-size:17px; color:#6E6E73; font-weight:400; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:26px; border-bottom:1px solid #EDEDEA; }
.stTabs [data-baseweb="tab"] { padding:0 2px 10px; font-size:14.5px; font-weight:450;
  color:#86868B; }
.stTabs [aria-selected="true"] { color:#1D1D1F; font-weight:590; }
.stTabs [data-baseweb="tab-highlight"] { background:#0071E3; height:2px; }

/* ---------- answer option rows (st.button in keyed containers) ---------- */
div.stButton > button {
  all:unset; box-sizing:border-box; display:flex; align-items:center; gap:12px;
  width:100%; padding:11px 14px; margin:0 0 8px;
  border:1px solid #E4E4E0; border-radius:10px; background:#FFFFFF;
  font:400 14.5px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  letter-spacing:-.008em; color:#1D1D1F; text-align:left; cursor:pointer;
  transition:border-color .12s ease, background .12s ease;
}
div.stButton > button:hover { border-color:#C9C9C4; background:#FAFAF8; }
div.stButton > button:focus-visible { outline:2px solid #0071E3; outline-offset:2px; }
div.stButton > button p { margin:0; font-size:inherit; }

[class*="st-key-optA"] div.stButton > button:before { content:"A"; }
[class*="st-key-optB"] div.stButton > button:before { content:"B"; }
[class*="st-key-optC"] div.stButton > button:before { content:"C"; }
[class*="st-key-optD"] div.stButton > button:before { content:"D"; }
[class*="st-key-opt"] div.stButton > button:before {
  flex:none; width:20px; height:20px;
  display:flex; align-items:center; justify-content:center;
  border-radius:5px; background:#F2F2EF; color:#6E6E73;
  font:600 11px ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
/* graded states via key suffix */
[class*="stateok"] div.stButton > button {
  border-color:#248A3D; background:rgba(36,138,61,.055); }
[class*="stateok"] div.stButton > button:before { background:#248A3D; color:#FFF; }
[class*="stateno"] div.stButton > button {
  border-color:#D70015; background:rgba(215,0,21,.045); }
[class*="stateno"] div.stButton > button:before { background:#D70015; color:#FFF; }
[class*="stateans"] div.stButton > button { border-color:#248A3D; background:#FFF; }
[class*="statemut"] div.stButton > button {
  border-color:#ECECE8; color:#A1A1A6; pointer-events:none; }
[class*="statemut"] div.stButton > button:before { background:#F7F7F4; color:#B8B8BD; }

/* primary + quiet buttons */
[class*="st-key-primary"] div.stButton > button { justify-content:center; width:auto;
  padding:11px 26px; border:none; border-radius:10px; background:#0071E3;
  color:#FFF; font-weight:590; }
[class*="st-key-primary"] div.stButton > button:before { content:none; }
[class*="st-key-primary"] div.stButton > button:hover { background:#0058B8; }
[class*="st-key-nextbtn"] div.stButton > button { justify-content:center; width:auto;
  padding:8px 20px; border:none; border-radius:9px; background:#1D1D1F;
  color:#FFF; font-size:13.5px; font-weight:590; }
[class*="st-key-nextbtn"] div.stButton > button:before { content:none; }
[class*="st-key-quiet"] div.stButton > button { width:auto; padding:8px 18px;
  font-size:13.5px; color:#6E6E73; }
[class*="st-key-quiet"] div.stButton > button:before { content:none; }
[class*="st-key-chip"] div.stButton > button { width:auto; padding:7px 14px;
  border-radius:9999px; font-size:13.5px; color:#48484D; background:#F7F7F4;
  border-color:#ECECE8; }
[class*="st-key-chip"] div.stButton > button:before { content:none; }

/* ---------- explanation ---------- */
.roger-why { font-size:14px; line-height:1.55; font-style:italic; color:#6E6E73;
  margin:18px 0 0; }
.roger-why b { font-style:normal; }

/* ---------- code block (st.code) ---------- */
[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre, .stCodeBlock pre {
  background:#FBFBF9; border:1px solid #E8E8E4; border-radius:10px;
  padding:14px 18px 14px 0;
}
[data-testid="stCode"] code, .stCodeBlock code {
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:13.5px; line-height:1.7; letter-spacing:-.002em; color:#1D1D1F;
}
[data-testid="stCode"] .linenos, [data-testid="stCode"] span[data-line-number] {
  display:inline-block; width:46px; padding-right:14px; margin-right:16px;
  text-align:right; color:#B8B8BD; background:#F7F7F4;
  border-right:1px solid #EFEFEB; user-select:none;
}
[data-testid="stCode"] button[title="Copy to clipboard"] { opacity:0; transition:.15s; }
[data-testid="stCode"]:hover button[title="Copy to clipboard"] { opacity:1; }
[data-testid="stCode"] .k, [data-testid="stCode"] .kn,
[data-testid="stCode"] .kd, [data-testid="stCode"] .ow { color:#9B2393; }
[data-testid="stCode"] .s, [data-testid="stCode"] .s1,
[data-testid="stCode"] .s2, [data-testid="stCode"] .sd { color:#C41A16; }
[data-testid="stCode"] .c, [data-testid="stCode"] .c1,
[data-testid="stCode"] .cm { color:#5D6C79; font-style:normal; }
[data-testid="stCode"] .m, [data-testid="stCode"] .mi,
[data-testid="stCode"] .mf { color:#1C00CF; }
[data-testid="stCode"] .nf, [data-testid="stCode"] .fm { color:#0F68A0; }
[data-testid="stCode"] .nc, [data-testid="stCode"] .nb { color:#3900A0; }

/* inline code in markdown */
.stMarkdown code { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:.88em; background:#F2F2EF; border:1px solid #E8E8E4;
  border-radius:5px; padding:1px 5px; color:#1D1D1F; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background:transparent; padding:0; margin:0 0 26px; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display:none; }
[class*="st-key-turnuser"] [data-testid="stChatMessage"] { display:flex;
  justify-content:flex-end; }
[class*="st-key-turnuser"] [data-testid="stChatMessageContent"] {
  max-width:76%; padding:10px 15px; border:1px solid #E9E9E5;
  border-radius:14px 14px 4px 14px; background:#F2F2EF;
  font-size:14.5px; line-height:1.5; letter-spacing:-.007em; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"] {
  font-size:14.5px; line-height:1.65; letter-spacing:-.007em; text-wrap:pretty; }
[class*="st-key-turnroger"] [data-testid="stChatMessageContent"]:before {
  content:"ROGER"; display:block; margin-bottom:10px;
  font:600 11.5px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  letter-spacing:.06em; color:#86868B; }
.roger-thinking { font-size:14.5px; color:#6E6E73;
  animation:rogerPulse 1.6s ease-in-out infinite; }
@keyframes rogerPulse { 0%,100%{opacity:.4} 50%{opacity:1} }

/* ---------- sources expander ---------- */
[data-testid="stExpander"] { border:none; border-top:1px solid #F0F0EE;
  border-radius:0; margin-top:6px; background:transparent; }
[data-testid="stExpander"] summary { padding:12px 0 0; font-size:12.5px;
  font-weight:590; color:#6E6E73; }
[data-testid="stExpander"] summary:hover { color:#1D1D1F; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding:10px 0 4px 16px; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p {
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:12.5px; line-height:1.9; color:#48484D; margin:0; }

/* ---------- progress ---------- */
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {
  font-size:13px; color:#6E6E73; letter-spacing:-.004em; margin-bottom:4px; }
.roger-reading { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:12px; color:#A1A1A6; margin-top:6px; }
[data-testid="stProgress"] > div > div { height:4px; background:#EAEAE7;
  border-radius:2px; }
[data-testid="stProgress"] > div > div > div { background:#0071E3;
  border-radius:2px; }

/* ---------- chat input pinned bottom ---------- */
[data-testid="stChatInput"] { border:1px solid #DEDEDA; border-radius:11px;
  background:#FBFBF9; padding:2px 4px; }
[data-testid="stChatInput"]:focus-within { border-color:#1D1D1F; background:#FFF; }
[data-testid="stChatInput"] textarea { font-size:14.5px; letter-spacing:-.006em;
  color:#1D1D1F; }
[data-testid="stChatInput"] textarea::placeholder { color:#A1A1A6; }
[data-testid="stBottomBlockContainer"] { background:#FFF;
  border-top:1px solid #F0F0EE; padding-bottom:18px; }
</style>
"""

# Pinned-light theme, applied per-process by the launcher (never written to
# the user's ~/.streamlit). From the design's config.toml deliverable.
THEME_ENV = {
    "STREAMLIT_THEME_BASE": "light",
    "STREAMLIT_THEME_PRIMARY_COLOR": "#0071E3",
    "STREAMLIT_THEME_BACKGROUND_COLOR": "#FFFFFF",
    "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": "#F7F7F4",
    "STREAMLIT_THEME_TEXT_COLOR": "#1D1D1F",
    "STREAMLIT_THEME_FONT": "sans-serif",
    "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
}
