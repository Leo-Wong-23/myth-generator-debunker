from __future__ import annotations

import json
import os
import random
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from streamlit_js_eval import streamlit_js_eval
from PIL import Image


# -----------------------------------------------------------------------------
# 0.  Predefined psychology topics for random selection
# -----------------------------------------------------------------------------

PSYCHOLOGY_TOPICS = [
    "cognitive bias",
    "growth mindset",
    "neuroplasticity", 
    "classical conditioning",
    "operant conditioning",
    "cognitive dissonance",
    "confirmation bias",
    "memory consolidation",
    "implicit bias",
    "emotional intelligence",
    "attachment theory",
    "social learning theory",
    "cognitive load theory",
    "positive psychology",
    "stereotype threat",
    "flow state",
    "learned helplessness",
    "placebo effect",
    "bystander effect",
    "fundamental attribution error",
    "working memory",
    "executive function",
    "theory of mind",
    "developmental psychology",
    "social psychology",
    "personality disorders",
    "stress response",
    "fight or flight response",
    "mirror neurons",
    "dual process theory",
    "cognitive behavioral therapy",
    "mindfulness",
    "resilience",
    "self-efficacy",
    "motivation theory",
    "decision making",
    "heuristics and biases",
    "psychological disorders",
    "trauma and PTSD",
    "addiction psychology"
]

# -----------------------------------------------------------------------------
# 1.  Low‑level conversation tree classes
# -----------------------------------------------------------------------------

@dataclass
class MsgNode:
    """A single message node in the conversation tree."""
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    parent_id: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ConvTree:
    """Tree of MsgNode objects with helpers for conversation branching and traversal."""

    def __init__(self):
        root_id = "root"
        self.nodes: Dict[str, MsgNode] = {
            root_id: MsgNode(id=root_id, role="system", content="ROOT", parent_id=None)
        }
        self.children: Dict[str, List[str]] = {root_id: []}
        self.current_leaf_id: str = root_id

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def add_node(self, parent_id: str, role: str, content: str) -> str:
        node_id = str(uuid.uuid4())
        node = MsgNode(id=node_id, role=role, content=content, parent_id=parent_id)
        self.nodes[node_id] = node
        self.children.setdefault(node_id, [])
        self.children.setdefault(parent_id, []).append(node_id)
        return node_id

    # ------------------------------------------------------------------
    # conversation branch & navigation helpers
    # ------------------------------------------------------------------
    def path_to_leaf(self, leaf_id: Optional[str] = None) -> List[MsgNode]:
        """Return the node path from root to *leaf_id* (defaults to current leaf)."""
        if leaf_id is None:
            leaf_id = self.current_leaf_id
        path: List[MsgNode] = []
        cursor = leaf_id
        while cursor is not None:
            path.append(self.nodes[cursor])
            cursor = self.nodes[cursor].parent_id
        return list(reversed(path))  # root ➜ leaf

    def siblings(self, node_id: str) -> List[str]:
        """Return the list of sibling node IDs for the given *node_id*."""
        parent_id = self.nodes[node_id].parent_id
        if parent_id is None:
            return []
        return self.children[parent_id]

    def sibling_index(self, node_id: str) -> int:
        """Return the index of *node_id* among its siblings, or -1 if not found."""
        sibs = self.siblings(node_id)
        return sibs.index(node_id) if node_id in sibs else -1

    def deepest_descendant(self, node_id: str) -> str:
        """Return the deepest descendant by always following the *first* child."""
        cursor = node_id
        while self.children.get(cursor):
            cursor = self.children[cursor][0]
        return cursor

    def select_sibling(self, node_id: str, direction: int) -> None:
        """Move *current_leaf_id* to the equivalent position on a sibling conversation branch."""
        sibs = self.siblings(node_id)
        if len(sibs) <= 1:
            return  # nothing to do
        idx = (self.sibling_index(node_id) + direction) % len(sibs)
        new_id = sibs[idx]
        # descend to the deepest leaf on that conversation branch
        self.current_leaf_id = self.deepest_descendant(new_id)

    # ------------------------------------------------------------------
    # Serialization helpers (for download transcript)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "nodes": {nid: node.__dict__ for nid, node in self.nodes.items()},
            "children": self.children,
            "current_leaf_id": self.current_leaf_id,
        }


# -----------------------------------------------------------------------------
# 2.  Environment & Gemini client
# -----------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
PASSWORD = os.getenv("PASSWORD")
client = genai.Client()


# -----------------------------------------------------------------------------
# 3.  Core generation & conversation helpers
# -----------------------------------------------------------------------------

def generate_myth_debunk_markdown(topic: str | None, n_misconceptions: int = 3) -> tuple[str, List[Tuple[str, str]]]:
    """
    Generate psychology myths and debunk them using Gemini API.
    Returns a tuple of (markdown-formatted string, list of source links).
    """
    system_instructions = textwrap.dedent(f'''
        You are a psychology myth generator and debunker.

        Find one psychology-related research finding, theory, or intervention. {' The focus topic is: ' + topic + '.' if topic else ''} Start by summarising and reporting the finding accurately. Next, create misunderstandings or misinterpretations of the finding, similar to the misconception generated from fMRI studies—like the myth that "we only use 2% of our brain," which sounds plausible but is false.

        # Task
        
        1. Find a solid psychology research finding, theory, or intervention.
        2. Summarise it elaborately and accurately.
        3. Write exactly {n_misconceptions} plausible but false misunderstandings** of that finding.
        4. For each misunderstanding, explain why it is misunderstood.

        # Steps

        1. **Research and Selection**: Identify a significant psychology-related finding, theory, or intervention from a reputable source.
        2. **Accurate Summary**: Provide a concise and accurate summary of the finding, highlighting key details and implications.
        3. **Misunderstanding Versions**: Develop interpretations or statements that misrepresent the original finding. They should sound plausible but are incorrect, similar to common myths or misconceptions.
        4. **Explanation**: Explain why the misunderstanding statements are wrong.

        # Output Format

        - **Summary**: A concise paragraph summarizing the psychology-related finding, theory, or intervention.
        - **Misunderstandings & Explanations**: List at least one or more plausible but false interpretations of the original finding.

        # Examples

        **Example 1:**

        - *Summary*: [Example summary on a psychological theory or finding, e.g., "Research on neuroplasticity has demonstrated that the brain can continually change and adapt throughout a person's life, forming new neural connections in response to learning and experience."]

        - *Misunderstandings*:
        1. "Since the brain can always change, everyone can become a genius if they just think hard enough about learning."
        2. "Neuroplasticity means trauma can be easily forgotten as the brain can simply rewire itself to erase bad memories."

        # Notes

        - Use credible sources when identifying psychology-related findings.
        - Ensure that summaries are factually correct and highlight the significance of the finding.
        - Aim for misunderstandings that are commonly heard or easily confused with actual scientific findings.
    ''').strip()

    # Configure Grounding tool for Google Search
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=1.0
    )
    
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
        contents=system_instructions,
        config=config,
    )

    # Extract grounding metadata from the first candidate
    grounding_meta = None
    if response.candidates:
        grounding_meta = response.candidates[0].grounding_metadata

    # Parse out the distinct source URLs and titles
    source_links: List[Tuple[str, str]] = []
    if grounding_meta and grounding_meta.grounding_chunks:
        for chunk in grounding_meta.grounding_chunks:
            # Try different possible attributes for the chunk
            url = None
            title = None
            
            # Check if chunk has web attribute
            if hasattr(chunk, 'web') and chunk.web:
                url = getattr(chunk.web, 'uri', None)
                title = getattr(chunk.web, 'title', url)
            # Fallback: check if chunk has direct url attribute
            elif hasattr(chunk, 'uri'):
                url = chunk.uri
                title = getattr(chunk, 'title', url)
            # Another fallback: check if chunk has url attribute
            elif hasattr(chunk, 'url'):
                url = chunk.url
                title = getattr(chunk, 'title', url)
            
            if url:
                # Use URL as title if no title is available
                display_title = title if title and title != url else url
                source_links.append((display_title, url))
    
    # Deduplicate while preserving order
    seen = set()
    source_links = [x for x in source_links if not (x[1] in seen or seen.add(x[1]))]

    return response.text, source_links


def build_system_message(myth_content: str = None) -> str:
    """Build system message for follow-up conversations about psychology myths."""
    base_msg = textwrap.dedent("""
        You are a psychology expert who specializes in debunking myths and misconceptions. 
        You can:
        - Explain psychological concepts in clear, accessible language
        - Provide evidence-based information about psychology research
        - Help clarify common misconceptions about psychological phenomena
        - Discuss the implications of psychological findings
        - Suggest related topics or follow-up questions
        
        Maintain an educational, supportive tone while being scientifically accurate.
    """).strip()
    
    if myth_content:
        context_msg = f"""
        
        IMPORTANT CONTEXT: Earlier in this session, you generated the following psychology content about myths and accurate findings:

        {myth_content}

        When users ask about "the myths you generated" or reference specific numbered myths, they are referring to the content above. Always refer back to this content when answering questions about the generated myths and findings.
        """
        return base_msg + context_msg
    
    return base_msg


def build_prompt(conv_tree: ConvTree, system_msg: str) -> str:
    """Build conversation prompt from conversation tree."""
    prompt_parts = [system_msg]
    
    for node in conv_tree.path_to_leaf()[1:]:  # skip root
        if node.role in {"user", "assistant"}:
            role_label = "User" if node.role == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {node.content}")
    
    return "\n\n".join(prompt_parts)


def generate_poster_image(topic: str, max_attempts: int = 3) -> Optional[Image.Image]:
    """
    Generate a poster image for the psychology myth topic using Imagen 4.
    Returns PIL Image if successful, None if failed after max_attempts.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            # Build a tailored prompt for the poster
            poster_prompt = f"""Create an educational poster about psychology concept: {topic}. 
            Style: Modern, scientific illustration with clean typography. 
            Include visual metaphors and symbols related to psychology and brain science. 
            Use a professional color palette with blues, greens, and white. 
            Make it suitable for educational materials."""
            
            # Configure image generation
            config = types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                person_generation="dont_allow",
                output_mime_type="image/png",
            )
            
            response = client.models.generate_images(
                model="models/imagen-4.0-generate-preview-06-06",
                prompt=poster_prompt,
                config=config,
            )
            
            # Convert GenAI SDK image object → raw bytes → PIL Image
            genai_img = response.generated_images[0].image
            raw = genai_img.image_bytes                   # bytes payload
            poster_img = Image.open(BytesIO(raw))          # now a proper PIL.Image.Image
            return poster_img
                
        except Exception as e:
            if attempt < max_attempts:
                st.warning(f"Poster generation attempt {attempt} failed: {e}. Retrying...")
            else:
                st.warning(f"Could not generate poster after {max_attempts} attempts. Last error: {e}")
    
    return None


def get_ai_response_for_myth(conv_tree: ConvTree, pending_user_node_id: str, myth_content: str) -> str:
    """Generate assistant reply for follow-up conversation about a specific myth."""
    system_msg = build_system_message(myth_content)
    conv_tree.current_leaf_id = pending_user_node_id
    
    # Build prompt string for Gemini
    prompt = build_prompt(conv_tree, system_msg)

    try:
        # Configure basic generation (no grounding needed for follow-up conversations)
        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=1500
        )
        
        response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt,
            config=config,
        )
        return response.text
    except genai.errors.ClientError as e:
        error_msg = f"Error: I encountered an API issue. Details: {e}\nPlease try again."
        return error_msg
    except Exception as e:
        error_msg = f"Error: Unexpected issue occurred. Details: {e}\nPlease try again."
        return error_msg


# -----------------------------------------------------------------------------
# 4.  Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Psychology Myth Generator-Debunker", layout="wide")

st.markdown(
    """
    <style>
        div.block-container{
            padding-top: 0rem !important;   /* remove the 2-rem gap everywhere */
        }
        .stTabs{                           /* keep the tab bar flush as well */
            margin-top: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Mobile / desktop switch ---
SCREEN_W = streamlit_js_eval(js_expressions="screen.width", key="w")
IS_MOBILE = bool(SCREEN_W and int(SCREEN_W) < 768)  # Bootstrap's "md" breakpoint

# --- Mobile-only CSS helper ---
if IS_MOBILE:
    st.markdown(
        """
        <style>
        /* Keep st.columns compact on screens ≤ 640 px */
        @media (max-width: 640px){

            /* 1 · remove the row-gap that forces wrapping */
            div[data-testid="horizontalBlock"]{
                row-gap:0 !important;
            }

            /* 2 · make each column shrink-to-fit */
            div[data-testid="horizontalBlock"] > div[data-testid="column"]{
                flex:0 0 auto !important;
                width:auto !important;
                padding-left:4px !important;
                padding-right:4px !important;
            }

            /* 3 · hide any empty row (prevents the blank band) */
            div[data-testid="horizontalBlock"]:not(:has(button,span)){
                display:none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

BUBBLE_MAX = "90%" if IS_MOBILE else "75%"

# ------------------ 3.1  Password gate ------------------

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Psychology Myth Generator-Debunker")
        st.markdown(
            """
            **Hi, this is Leo, a psychology and cognitive neuroscience postgraduate with backgrounds in AI and education. Welcome to this Psychology Myth Generator-Debunker that I built!**

            This is a proof-of-concept application to explore how AI can bring educational innovations to the field of psychology. This app is designed to help students, educators, and psychology enthusiasts understand psychological research by generating accurate findings alongside common myths and misconceptions. Specifically for trainee psychologists, this app facilitates their skills in detecting misconceptions and explaining them to clients and the general public.<br><br>

            **Key Features:**
            - Generate accurate psychology research findings with evidence-based summaries
            - Optionally focus on specific topics or explore general psychology concepts
            - Create and debunk common myths and misconceptions about psychological phenomena
            - Interactive follow-up conversations to explore topics in greater depth<br><br>

            ***Safety & Privacy Statement:***
            This app is currently in development and serves as an educational demonstration tool. 
            No chat history or personal data are stored beyond the active session; they are erased once you close or refresh the page.

            Please enter the password to begin (you can find it in my CV).
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="password_form"):
            entered = st.text_input("Enter Password:", type="password")
            if st.form_submit_button("Submit"):
                if entered == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Try again.")

    if not st.session_state.authenticated:
        st.stop()

check_password()


# ---------------------------------------------------------------------------
# Helper: render one message bubble for myth-specific conversations
# ---------------------------------------------------------------------------
def render_msg_for_myth(node: MsgNode, myth_conv_tree: ConvTree, myth_id: str, mobile: bool = False):
    # ----- generic style helpers ------------------------------------------
    role_label = (
        "Psychology Expert" if node.role == "assistant"
        else "You"
    )
    align        = "flex-start" if node.role == "assistant" else "flex-end"
    bubble_color = "#1b222a"     if node.role == "assistant" else "#0e2a47"
    text_color   = "white"

    # put most of the "large-screen tweakery" behind the mobile switch
    if mobile:
        OFFSET_TOP            = "8px"
        OFFSET_TOP_INDICATOR  = "8px"
        TRANSFORM             = "none"
        TRANSFORM_INDICATOR   = "none"
        TRANSFORM_LB          = "none"
    else:
        OFFSET_TOP            = "25px"
        OFFSET_TOP_INDICATOR  = "32px"
        TRANSFORM             = "translate(-80px, 0)"
        TRANSFORM_INDICATOR   = "translate(10px, 0)"
        TRANSFORM_LB          = "none"

    BUBBLE_MAX = "90%" if mobile else "75%"

    # ------------------------------------------------------------------
    # 1) "edit mode" for the user's own message
    # ------------------------------------------------------------------
    edit_key = f"editing_msg_id_{myth_id}"
    edit_content_key = f"editing_content_{myth_id}"
    
    if edit_key not in st.session_state:
        st.session_state[edit_key] = None
    if edit_content_key not in st.session_state:
        st.session_state[edit_content_key] = ""
    
    if node.role == "user" and st.session_state[edit_key] == node.id:
        new_text = st.text_area(
            "Edit your message:",
            value=st.session_state[edit_content_key] or node.content,
            key=f"textarea_{myth_id}_{node.id}",
        )
        col_l, col_cancel, col_send, col_r = st.columns([6, 1, 1, 6], gap="small")

        with col_cancel:
            if st.button("Cancel", key=f"cancel_edit_{myth_id}_{node.id}"):
                st.session_state[edit_key] = None
                st.session_state[edit_content_key] = ""
                st.rerun()

        with col_send:
            if st.button("Send", key=f"send_edit_{myth_id}_{node.id}"):
                parent = node.parent_id                     # ⬅ current branch root
                new_user_id = myth_conv_tree.add_node(parent, "user", new_text)
                
                # Get myth content for this specific myth
                myth_content = next((tab["content"] for tab in st.session_state.myth_tabs if tab["id"] == myth_id), "")
                
                with st.spinner("Expert is responding…"):
                    ai_reply = get_ai_response_for_myth(myth_conv_tree, new_user_id, myth_content)
                new_assist_id = myth_conv_tree.add_node(new_user_id, "assistant", ai_reply)
                myth_conv_tree.current_leaf_id = new_assist_id
                st.session_state[edit_key] = None
                st.session_state[edit_content_key] = ""
                st.rerun()
        return  # don't fall through to normal rendering

    # ------------------------------------------------------------------
    # 2) User messages (with or without version controls)
    # ------------------------------------------------------------------
    if node.role == "user":
        sibs          = myth_conv_tree.siblings(node.id)
        has_versions  = len(sibs) > 1

        # -------------------- 2a. *With* (re-)version controls ------------
        if has_versions:
            idx    = myth_conv_tree.sibling_index(node.id) + 1
            total  = len(sibs)

            # ====== PHONE LAYOUT ===============================
            if mobile:
                # Inject the CSS once per session
                css_key = f"mob_css_injected_{myth_id}"
                if css_key not in st.session_state:
                    st.markdown(
                        """
                        <style>
                        /* tiny flex row that survives Streamlit's mobile stacking  */
                        .mobile-ctrls{
                            display:flex; align-items:center; gap:6px;
                            margin-top:8px; margin-bottom:2px;
                        }
                        .mobile-ctrls button[kind="secondary"]{
                            width:34px!important; height:34px!important;
                            padding:2px 4px!important; font-size:18px!important;
                        }
                        .mobile-ctrls span.ver{
                            min-width:38px; text-align:center; font-size:16px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state[css_key] = True

                # ---- flex container for ◀ 1/2 ▶ ✏️ ----------------------
                with st.container():
                    st.markdown('<div class="mobile-ctrls">', unsafe_allow_html=True)

                    # ◀ previous
                    if st.button("◀", key=f"left_{myth_id}_{node.id}"):
                        myth_conv_tree.select_sibling(node.id, -1)
                        st.rerun()

                    # version badge
                    st.markdown(f"<span class='ver'>{idx}/{total}</span>",
                                unsafe_allow_html=True)

                    # ▶ next
                    if st.button("▶", key=f"right_{myth_id}_{node.id}"):
                        myth_conv_tree.select_sibling(node.id, +1)
                        st.rerun()

                    # ✏️ edit
                    if st.button("✏️", key=f"edit_{myth_id}_{node.id}"):
                        st.session_state[edit_key] = node.id
                        st.session_state[edit_content_key] = node.content
                        st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

                # ----- finally the message bubble itself -----------------
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:{align}; margin:4px 0 12px;'>
                      <div style='background-color:{bubble_color}; color:{text_color};
                                  padding:12px 16px; border-radius:18px;
                                  max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{node.content}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return  # phone version done

            # ====== DESKTOP LAYOUT =========
            col_left, col_center, col_right, col_edit, col_bubble = st.columns(
                [1.5, 1.5, 2, 6, 40], gap="small"
            )

            # ◀
            with col_left:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM_LB};'>",
                    unsafe_allow_html=True)
                if st.button("◀", key=f"left_{myth_id}_{node.id}"):
                    myth_conv_tree.select_sibling(node.id, -1); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # 1/2
            with col_center:
                st.markdown(
                    f"""
                    <div style='display:flex; align-items:center; margin-top:{OFFSET_TOP_INDICATOR};
                                transform:{TRANSFORM_INDICATOR};'>{idx}/{total}</div>
                    """,
                    unsafe_allow_html=True)

            # ▶
            with col_right:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM};'>",
                    unsafe_allow_html=True)
                if st.button("▶", key=f"right_{myth_id}_{node.id}"):
                    myth_conv_tree.select_sibling(node.id, +1); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Edit
            with col_edit:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM};'>",
                    unsafe_allow_html=True)
                if st.button("Edit Message", key=f"edit_{myth_id}_{node.id}"):
                    st.session_state[edit_key] = node.id
                    st.session_state[edit_content_key] = node.content
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Bubble
            with col_bubble:
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                      <div style='background-color:{bubble_color}; color:{text_color};
                                  padding:12px 16px; border-radius:18px;
                                  max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{node.content}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return  # end "user w/versions" path

        # -------------------- 2b. user message WITHOUT versions --------
        if mobile:
            edit_col, bubble_col = st.columns([1, 10], gap="small")
            edit_label = "✏️"
        else:
            edit_col, bubble_col = st.columns([1, 9], gap="small")
            edit_label = "Edit Message"

        with edit_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP};'>",
                unsafe_allow_html=True)
            if st.button(edit_label, key=f"edit_{myth_id}_{node.id}"):
                st.session_state[edit_key] = node.id
                st.session_state[edit_content_key] = node.content
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with bubble_col:
            st.markdown(
                f"""
                <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                  <div style='background-color:{bubble_color}; color:{text_color};
                              padding:12px 16px; border-radius:18px;
                              max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                              font-size:16px; line-height:1.5;'>
                    <strong>{role_label}:</strong><br>{node.content}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return  # end "user w/o versions" path

    # ------------------------------------------------------------------
    # 3) Assistant (psychology expert) bubble
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div style='display:flex; justify-content:{align}; margin:8px 0;'>
          <div style='background-color:{bubble_color}; color:{text_color};
                      padding:12px 16px; border-radius:18px;
                      max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                      font-size:16px; line-height:1.5;'>
            <strong>{role_label}:</strong><br>{node.content}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------ 3.2  Session‑state bootstrapping ------------------

# Dynamic tabs for myths
if "myth_tabs" not in st.session_state:
    st.session_state.myth_tabs = []  # List of myth data for each tab: [{"id": "myth_1", "content": "...", "title": "..."}]

if "myth_conversations" not in st.session_state:
    st.session_state.myth_conversations = {}  # Dict mapping myth_id to ConvTree

if "myth_counter" not in st.session_state:
    st.session_state.myth_counter = 0  # Counter for unique myth IDs

# Helper function to create a new myth tab
def create_new_myth_tab(myth_content: str, source_links: List[Tuple[str, str]], topic: str = None, n_myths: int = 3, poster_image: Optional[Image.Image] = None) -> str:
    """Create a new myth tab and return its ID"""
    st.session_state.myth_counter += 1
    myth_id = f"myth_{st.session_state.myth_counter}"
    
    # Extract title from content (first line or use default)
    title_lines = myth_content.split('\n')[:3]  # First few lines to find a title
    title = f"Set {st.session_state.myth_counter} - Myth Set"
    for line in title_lines:
        if line.strip() and not line.startswith('#'):
            # Try to find topic or create a descriptive title
            if topic:
                title = f"Set {st.session_state.myth_counter} - {topic}"
            else:
                title = f"Set {st.session_state.myth_counter} - Myths"
            break
    
    # Create myth tab data
    myth_tab = {
        "id": myth_id,
        "content": myth_content,
        "source_links": source_links,
        "title": title,
        "topic": topic,
        "n_myths": n_myths,
        "poster_image": poster_image
    }
    
    # Add to tabs list
    st.session_state.myth_tabs.append(myth_tab)
    
    # Create new conversation tree for this myth
    st.session_state.myth_conversations[myth_id] = ConvTree()
    
    return myth_id

def generate_combined_transcript() -> str:
    """Generate a combined transcript for all myth sets and their conversations."""
    if not st.session_state.myth_tabs:
        return "No myth sets available."
    
    lines = []
    lines.append("=" * 80)
    lines.append("PSYCHOLOGY MYTH GENERATOR-DEBUNKER")
    lines.append("COMBINED TRANSCRIPT FOR ALL MYTH SETS")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    for i, myth_tab in enumerate(st.session_state.myth_tabs, 1):
        myth_id = myth_tab["id"]
        myth_content = myth_tab["content"]
        myth_conv_tree = st.session_state.myth_conversations.get(myth_id)
        
        # Add myth set header
        lines.append(f"\n\n{'=' * 60}")
        lines.append(f"MYTH SET {i}: {myth_tab['title']}")
        if myth_tab.get('topic'):
            lines.append(f"Topic: {myth_tab['topic']}")
        lines.append(f"{'=' * 60}")
        
        # Add generated content
        lines.append("\n--- GENERATED CONTENT ---")
        lines.append(myth_content)
        
        # Add conversation if exists
        if myth_conv_tree:
            conversation_nodes = myth_conv_tree.path_to_leaf()[1:]
            if conversation_nodes:
                lines.append("\n--- CONVERSATION ---")
                for node in conversation_nodes:
                    if node.role in {"user", "assistant"}:
                        role_name = "Psychology Expert" if node.role == "assistant" else "User"
                        lines.append(f"\n{role_name}:")
                        lines.append(node.content)
    
    lines.append(f"\n\n{'=' * 80}")
    lines.append("END OF COMBINED TRANSCRIPT")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# Trace‑back editing state
if "editing_msg_id" not in st.session_state:
    st.session_state.editing_msg_id = None
if "editing_content" not in st.session_state:
    st.session_state.editing_content = ""
if "pending_user_node_id" not in st.session_state:
    st.session_state.pending_user_node_id = None

# Active tab state
if "active_tab" not in st.session_state:
    # default to App Information
    st.session_state.active_tab = "App Information"

# ------------------ 3.3  Sidebar (now empty) ------------------

# Sidebar is now empty - all controls moved to appropriate tabs


# ------------------ 3.4  Dynamic Tabs ------------------

# Build tab list dynamically
tab_labels = ["App Information"]

# Add tabs for each generated myth
for myth_tab in st.session_state.myth_tabs:
    tab_labels.append(f"🧠 {myth_tab['title']}")

# Add "Generate New Myth" tab
tab_labels.append("➕ Generate New Myth")

# Create tabs
tabs = st.tabs(tab_labels)

# 🔄 If we just generated a myth, simulate a click on the *second-last* tab
if st.session_state.get("just_generated"):
    streamlit_js_eval(
        js_expressions=textwrap.dedent("""
            const btns = window.parent.document
                        .querySelectorAll('button[data-baseweb="tab"]');
            if (btns.length > 1) {
                btns[btns.length - 2].click();
            }
        """),
        key="auto_switch_tab",
    )
    st.session_state.pop("just_generated")


# TAB 1  ── App Information ----------------------------------------------
with tabs[0]:  # App Information tab
    st.header("🧠 Psychology Myth Generator-Debunker")
    
    st.subheader("Purpose & Educational Value")
    st.markdown("""
    This application serves as an educational tool to help psychology students, educators, and enthusiasts:
    
    - **Distinguish Science from Pseudoscience**: Learn to identify the difference between rigorous psychological research and popular misconceptions
    - **Develop Critical Thinking**: Practice evaluating claims about psychological phenomena with a scientific mindset
    - **Understand Research Interpretation**: See how legitimate findings can be misinterpreted or overgeneralized
    - **Explore Psychology Topics**: Generate content about specific areas of interest or discover new psychological concepts
    """)
    
    st.subheader("How to Use This App")
    st.markdown("""
    1. **Generate New Myths**: Use the "Generate New Myth" tab to create sets of psychology myths and accurate findings
    2. **Set Parameters**: Select the number of myths and optional focus topic to tailor the content
    3. **Optional Poster Generation**: Check the "Generate poster" option to create an educational poster using AI image generation
    4. **Explore Myths**: Each generated myth set gets its own tab for focused exploration
    5. **Follow-Up Conversations**: Use the chat feature in each myth tab to ask follow-up questions about the psychology concepts
    """)

    st.subheader("How This App Works")
    st.markdown("""
    - The app uses Google's Gemini 2.5 Flash Lite model via their application programming interface (API) to generate responses based on user-defined parameters
    - Each time you click "Generate New Myth", the AI model searches on the internet using Google Search grounding for psychology-related findings, theories, or interventions, focusing on a specific topic if provided
    - Based on its search results, the AI summarises the finding and generates plausible but false misunderstandings or myths about it
    - When the poster option is enabled, the app uses Google's Imagen 4 model to generate educational posters related to the psychology topic
    """)


# Dynamic Myth Tabs (one for each generated myth set)
for i, myth_tab in enumerate(st.session_state.myth_tabs):
    with tabs[i + 1]:  # +1 because first tab is App Information
        myth_id = myth_tab["id"]
        myth_content = myth_tab["content"]
        
        # Display poster image if available
        if myth_tab.get("poster_image"):
            st.subheader("🎨 Educational Poster")
            # Use responsive width: fit screen width on mobile, 400px on desktop
            poster_width = None if IS_MOBILE else 400
            st.image(myth_tab["poster_image"], caption=f"Poster for: {myth_tab.get('topic', 'Psychology Concepts')}", width=poster_width)
            st.markdown("---")
        
        # Display the myth content
        st.subheader("🔍 Generated Findings & Myths")
        st.markdown(myth_content)
        
        # Display source links if available
        if myth_tab.get("source_links"):
            with st.expander("🔗 Show Google Search grounding source links"):
                for title, url in myth_tab["source_links"]:
                    # Show both title and full URL
                    if title and title != url:
                        st.markdown(f"- **{title}**  \n  [{url}]({url})")
                    else:
                        st.markdown(f"- [{url}]({url})")
        else:
            with st.expander("🔗 Show Google Search grounding source links"):
                st.write("No links available.")
        
        st.markdown("---")
        st.subheader("💬 Follow-Up Conversation")
        st.markdown("Ask follow-up questions about the psychology concepts above:")

        # Get the conversation tree for this specific myth
        myth_conv_tree = st.session_state.myth_conversations[myth_id]

        # Display conversation history for this myth
        for node in myth_conv_tree.path_to_leaf()[1:]:
            render_msg_for_myth(node, myth_conv_tree, myth_id, mobile=IS_MOBILE)

        # Handle pending AI response for this myth
        pending_key = f"pending_user_node_id_{myth_id}"
        if pending_key not in st.session_state:
            st.session_state[pending_key] = None
            
        if st.session_state[pending_key]:
            with st.spinner("Waiting for Response..."):
                ai_reply = get_ai_response_for_myth(
                    myth_conv_tree,
                    st.session_state[pending_key],
                    myth_content
                )
            new_assist = myth_conv_tree.add_node(
                st.session_state[pending_key], "assistant", ai_reply
            )
            myth_conv_tree.current_leaf_id = new_assist
            st.session_state[pending_key] = None
            st.rerun()

        # Chat input for this specific myth
        user_text = st.chat_input(f"Ask about the psychology concepts above...", key=f"chat_input_{myth_id}")
        if user_text:
            new_user = myth_conv_tree.add_node(myth_conv_tree.current_leaf_id, "user", user_text)
            myth_conv_tree.current_leaf_id = new_user
            st.session_state[pending_key] = new_user
            st.rerun()

        # Management buttons
        # Download transcript for this myth - generate content directly in download button
        lines = []
        lines.append("=== GENERATED CONTENT ===")
        lines.append(myth_content)
        
        # Add conversation section if there are any messages
        conversation_nodes = myth_conv_tree.path_to_leaf()[1:]
        if conversation_nodes:
            lines.append("\n=== CONVERSATION ===")
            for node in conversation_nodes:
                if node.role in {"user", "assistant"}:
                    role_name = "Psychology Expert" if node.role == "assistant" else "User"
                    lines.append(f"{role_name}: {node.content}")
        
        transcript = "\n\n".join(lines)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown("---")
        st.download_button(
            f"📥 Download Transcript",
            data=transcript,
            file_name=f"psychology_myths_{myth_id}_{ts}.txt",
            mime="text/plain",
            key=f"download_btn_{myth_id}"
        )
        
        # Delete this myth tab
        if st.button(f"🗑️ Delete This Myth Set", key=f"delete_{myth_id}"):
            # Remove from tabs list
            st.session_state.myth_tabs = [tab for tab in st.session_state.myth_tabs if tab["id"] != myth_id]
            # Remove conversation
            if myth_id in st.session_state.myth_conversations:
                del st.session_state.myth_conversations[myth_id]
            # Remove pending state
            if pending_key in st.session_state:
                del st.session_state[pending_key]
            st.rerun()


# Last Tab: Generate New Myth
with tabs[-1]:  # Last tab is always "Generate New Myth"
    st.subheader("Generate Psychology Myth Set")
    
    topic_mode = st.radio(
        "**Topic Selection**",
        options=["🎲 Random", "📋 Select from list", "✏️ Enter custom topic"],
        key="topic_mode",
        horizontal=True
    )
    
    topic = None
    
    if topic_mode == "🎲 Random":
        if st.button("**Generate Topic**", key="random_topic_btn"):
            random_topic = random.choice(PSYCHOLOGY_TOPICS)
            st.session_state["selected_random_topic"] = random_topic
        
        if "selected_random_topic" in st.session_state:
            topic = st.session_state["selected_random_topic"]
            st.success(f"🎯 Selected topic: **{topic}**")
    
    elif topic_mode == "📋 Select from list":
        topic = st.selectbox(
            "Choose a psychology topic:",
            options=[""] + PSYCHOLOGY_TOPICS,
            key="topic_selectbox",
            help="Select a predefined psychology topic to focus on"
        )
    
    elif topic_mode == "✏️ Enter custom topic":
        topic = st.text_input(
            'Enter your custom topic:',
            key="custom_topic_input",
            placeholder="e.g., cognitive bias, growth mindset, etc.",
            max_chars=100,
        )
    
    # Generation Parameters Section
    n_myths = st.slider('**Number of misconceptions**', min_value=1, max_value=10, value=3, key="new_myth_count")
    
    # Poster generation option
    poster_enabled = st.toggle(
        "🎨 Generate poster for this myth", 
        value=False,
        help="When enabled, generate a themed poster above your myth using Imagen 4",
        key="poster_enabled"
    )
    
    # Generate button - only enabled if topic is selected/entered
    can_generate = bool(topic and topic.strip())
    
    st.markdown("---")

    if st.button("**✨ Generate New Myth Set**", key="generate_new_myth", disabled=not can_generate):
        with st.spinner("Generating new psychology myths and debunking..."):
            try:
                myth_content, source_links = generate_myth_debunk_markdown(topic, n_myths)
                
                # Generate poster if enabled
                poster_image = None
                if poster_enabled:
                    with st.spinner("Generating poster..."):
                        poster_image = generate_poster_image(topic)
                
                myth_id = create_new_myth_tab(myth_content, source_links, topic, n_myths, poster_image)

                # Flag the next rerun so we can auto-switch tabs
                st.session_state["just_generated"] = True

                st.rerun()
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
    
    # Show existing myth sets
    if st.session_state.myth_tabs:
        st.markdown("---")
        st.subheader("📚 Existing Myth Sets")
        for i, myth_tab in enumerate(st.session_state.myth_tabs):
            st.markdown(f"**{myth_tab['title']}**")
        
        # Download transcript for all myth sets button - generate content directly
        combined_transcript = generate_combined_transcript()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download Transcript for All Myth Sets",
            data=combined_transcript,
            file_name=f"psychology_myths_combined_{ts}.txt",
            mime="text/plain",
            key="download_all_btn"
        )
        
        # Clear all myths button at the bottom
        if st.button("🗑️ Clear All Myths", key="clear_all_myths"):
            st.session_state.myth_tabs = []
            st.session_state.myth_conversations = {}
            st.session_state.myth_counter = 0
            # Clear all pending states
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("pending_user_node_id_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.rerun()

# -----------------------------------------------------------------------------
# 4.  __main__ guard
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
