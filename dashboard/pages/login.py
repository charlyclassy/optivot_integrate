"""Sign-in and create-account screens."""
from __future__ import annotations

import streamlit as st

import config
from dashboard import database
from dashboard import theme as ui

AUTH_CSS = """
<style>
[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], [data-testid='stExpandSidebarButton'] { display:none !important; }
.block-container { max-width: 1120px; padding-top: 5vh; }
.st-key-auth_brand { background: radial-gradient(700px 400px at 100% 0%, rgba(34,211,238,.25), transparent 60%),
  linear-gradient(150deg, #0b1020 0%, #1e1b4b 60%, #3730a3 100%); border-radius:26px; padding:44px 40px; min-height:560px; }
.au-brand { display:flex; align-items:center; gap:10px; font-size:20px; font-weight:700; color:#fff; letter-spacing:-0.02em; }
.au-h { font-size:34px; font-weight:800; line-height:1.12; color:#fff; letter-spacing:-0.035em; margin:44px 0 12px 0; }
.au-p { font-size:15px; line-height:1.6; color:#cbd5e1; margin-bottom:26px; }
.au-point { display:flex; gap:12px; align-items:flex-start; margin:14px 0; color:#e2e8f0; font-size:14.5px; line-height:1.45; }
.au-point .ic { width:28px; height:28px; border-radius:9px; background:rgba(255,255,255,.10); color:#a5b4fc; display:flex;
  align-items:center; justify-content:center; flex:0 0 28px; }
.st-key-auth_form { padding:26px 10px 0 26px; }
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 340px); }
.au-h { font-size:clamp(26px, 3vw, 34px) !important; }
@media (max-width: 820px) {
  .st-key-auth_brand { min-height:0 !important; padding:26px 24px; }
  .au-h { margin-top:22px; }
  .au-point { display:none; }
  .st-key-auth_form { padding:10px 2px 0 2px; }
}
.au-title { font-size:30px; font-weight:800; letter-spacing:-0.03em; color:#0f172a; margin:0; line-height:1.2; }
.au-sub { font-size:14.5px; color:#475569; margin:6px 0 18px 0; }
.au-role { font-size:13px; line-height:1.55; color:#475569; background:#eef2ff; border:1px solid #e0e7ff; border-radius:12px;
  padding:10px 14px; margin-bottom:14px; }
.au-role b { color:#3730a3; }
.au-foot { font-size:12.5px; color:#94a3b8; margin-top:16px; }
</style>
"""
POINTS = [
    ("trend", "Forecasts for every depot from a saved Optivolt LSTM model"),
    ("alert", "Warnings and breach alerts with lead time"),
    ("sliders", "Scenario analysis for fleet growth and capacity"),
    ("file", "Reports and a history of forecasts against outcomes"),
]


def _go(view: str) -> None:
    """Navigate to the registered public page, rather than resetting a view flag.

    Each public page is a separate st.Page. Mutating public_view alone cannot
    change the selected Streamlit page, so the previous Create a workspace
    button appeared to do nothing.
    """
    pages = st.session_state.get("_public_nav_pages", {})
    target = pages.get(view)
    if target is None:
        st.error("The requested page is unavailable. Please refresh the application.")
        return
    st.switch_page(target)


def _sign_in_user(user: dict, remember: bool) -> None:
    st.session_state["user"] = user
    st.session_state.pop("public_view", None)
    if remember:
        token = database.create_session(user["email"])
        st.session_state["session_token"] = token
        st.query_params["session"] = token
    st.rerun()


def _brand_panel() -> None:
    with st.container(key="auth_brand"):
        st.markdown(
            f'<div class="au-brand">{ui.LOGO_SVG.format(size=38)}{ui.esc(config.PRODUCT_NAME)}</div>'
            '<div class="au-h">Stay ahead of your depot\'s capacity risk</div>'
            '<div class="au-p">Forecast EV charging demand, see capacity risk early and plan fleet growth with confidence.</div>'
            + "".join(f'<div class="au-point"><span class="ic">{ui.icon(name, 15)}</span><span>{ui.esc(text)}</span></div>'
                      for name, text in POINTS),
            unsafe_allow_html=True)


def _sign_in_form() -> None:
    st.markdown('<div class="au-title">Welcome back</div><div class="au-sub">Sign in to your account to continue.</div>',
                unsafe_allow_html=True)
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    with st.form("login", border=False):
        email = st.text_input("Email or username", placeholder="name@company.com", autocomplete="username")
        password = st.text_input("Password", type="password", placeholder="Your password",
                                 autocomplete="current-password")
        remember = st.checkbox("Remember me on this device", value=True)
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        try:
            user = database.authenticate(email, password)
        except database.AccountError as exc:  # locked account or suspended organisation
            user = None
            st.error(str(exc), icon=":material/error:")
        else:
            if user is None:
                st.error("The email or password is not correct, or the account is inactive.", icon=":material/error:")
        if user is not None:
            _sign_in_user(user, remember)
    with st.popover("Forgot password?", width="stretch"):
        st.markdown("**Reset your password**")
        st.write("Ask your organisation's administrator to set a new password for you under Settings > Users. "
                 "Once signed in, you can change it yourself under Settings > My account.")
    left, right = st.columns(2)
    if database.signup_allowed():
        left.button("Create a workspace", key="to_signup", on_click=_go, args=("signup",), width="stretch")
    right.button("Back to home", key="to_home", on_click=_go, args=("landing",), width="stretch", type="tertiary")


def _sign_up_form() -> None:
    st.markdown('<div class="au-title">Create your workspace</div>'
                '<div class="au-sub">Set up your organisation and start forecasting in a minute.</div>',
                unsafe_allow_html=True)
    if not database.signup_allowed():
        st.info("New sign-ups are currently closed. Ask your organisation's administrator to add you, "
                "or contact us for access.", icon=":material/lock:")
        st.button("Back to sign in", on_click=_go, args=("signin",))
        return
    st.markdown('<div class="au-role">You become the <b>Administrator</b> of a new, private workspace for your '
                'organisation. You can then add colleagues as <b>Administrators</b> or read-only <b>Viewers</b> '
                'under Settings &gt; Users.</div>', unsafe_allow_html=True)
    with st.form("signup", border=False):
        organisation = st.text_input("Organisation name", placeholder="Acme Fleet Services", autocomplete="organization")
        a, b = st.columns(2)
        name = a.text_input("Full name", placeholder="Alex Morgan", autocomplete="name")
        email = b.text_input("Work email", placeholder="name@company.com", autocomplete="email")
        a, b = st.columns(2)
        password = a.text_input("Password", type="password", autocomplete="new-password",
                                help=f"At least {database.MIN_PASSWORD_LENGTH} characters, with letters and numbers or symbols.")
        confirm = b.text_input("Confirm password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Create workspace", type="primary", width="stretch")
    if submitted:
        if password != confirm:
            st.error("The two passwords do not match.", icon=":material/error:")
        else:
            try:
                user = database.create_organisation(organisation, email, name, password)
            except database.AccountError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                _sign_in_user(user, remember=True)
    left, right = st.columns(2)
    left.button("I already have an account", key="to_signin", on_click=_go, args=("signin",), width="stretch")
    right.button("Back to home", key="to_home2", on_click=_go, args=("landing",), width="stretch", type="tertiary")


def render(view: str = "signin") -> None:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    left, right = st.columns([1, 1.05], gap="large")
    with left:
        _brand_panel()
    with right:
        with st.container(key="auth_form"):
            if view == "signup":
                _sign_up_form()
            else:
                _sign_in_form()
            st.markdown(f'<div class="au-foot">{ui.esc(config.PRODUCT_NAME)} &middot; Secure access for authorised users only</div>',
                        unsafe_allow_html=True)
