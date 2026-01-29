"""
Visualization module for Patent Landscape Map.
Effectively visualizes the relationship between User Idea and Search Results.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

def render_patent_map(result: dict):
    """
    Render a premium interactive Patent Landscape Map.
    
    Features:
    - Quadrant analysis (Direct Risk, Technical Reference, Noise, Strategic Avoidance)
    - User Idea Center Point
    - Distance-based conceptual alignment visualization
    """
    search_results = result.get('search_results', [])
    user_idea = result.get('user_idea', '내 아이디어')
    
    if not search_results:
        st.caption("시각화할 데이터가 충분하지 않습니다.")
        return

    # Prepare data for DataFrame
    data = []
    
    # 1. Add User Idea as the Origin/Goal point
    data.append({
        "Patent ID": "🎯 My Idea",
        "Title": "내 아이디어 (분석 기준점)",
        "Conceptual Alignment": 1.0,  # Center point for idea
        "Analytical Depth": 1.0,
        "Relevance": 15,
        "Category": "My Idea",
        "Abstract": user_idea[:200]
    })
    
    # 2. Add search results
    for r in search_results:
        # We use grading_score for alignment and score/stats for depth
        alignment = r.get('grading_score', 0.5)
        # Combine dense and sparse scores for depth (normalized roughly)
        depth = (r.get('dense_score', 0) * 0.7 + min(r.get('sparse_score', 0) / 50, 1.0) * 0.3)
        
        grade = r.get('grading_score', 0)
        
        if grade >= 0.75:
            cat = "🚨 침해 주의 (高)"
        elif grade >= 0.5:
            cat = "🟡 기술적 참고"
        elif alignment > 0.6 and depth < 0.4:
            cat = "🕵️ 숨겨진 경쟁자"
        else:
            cat = "📗 단순 키워드 중복"
            
        data.append({
            "Patent ID": r.get('patent_id'),
            "Title": r.get('title'),
            "Conceptual Alignment": alignment,
            "Analytical Depth": depth,
            "Relevance": grade * 25 + 5,
            "Category": cat,
            "Abstract": r.get('abstract', '')[:150] + "..."
        })
        
    df = pd.DataFrame(data)
    
    # Create Scatter Plot with premium styling
    fig = px.scatter(
        df,
        x="Conceptual Alignment",
        y="Analytical Depth",
        size="Relevance",
        color="Category",
        hover_name="Title",
        hover_data={"Patent ID": True, "Abstract": True, "Relevance": False},
        color_discrete_map={
            "My Idea": "#00d4ff",
            "🚨 침해 주의 (高)": "#ff4b4b",
            "🟡 기술적 참고": "#ffa500",
            "🕵️ 숨겨진 경쟁자": "#6c5ce7",
            "📗 단순 키워드 중복": "#a0a0a0"
        },
        title="✨ Premium Patent Landscape Analysis",
        template="plotly_white"
    )
    
    # Ivory background color (#fdfaf5) to match the app theme
    ivory_bg = "#fdfaf5"
    grid_color = "rgba(0,0,0,0.1)"
    line_color = "rgba(0,0,0,0.2)"
    
    # Add Quadrant Backgrounds/Annotations using shapes if possible, or just layout lines
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color=line_color)
    fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color=line_color)
    
    fig.update_layout(
        xaxis_title="🎯 기술적 정렬도 (Conceptual Alignment)",
        yaxis_title="🔍 분석 심도 (Analytical Depth)",
        legend_title="Risk & Value",
        hovermode="closest",
        height=660, # Increased height to accommodate axis descriptions
        margin=dict(l=60, r=60, t=100, b=120), # Increased bottom margin
        plot_bgcolor=ivory_bg,
        paper_bgcolor=ivory_bg,
        xaxis=dict(range=[-0.1, 1.1], gridcolor=grid_color),
        yaxis=dict(range=[-0.1, 1.1], gridcolor=grid_color),
        font=dict(family="Pretendard, sans-serif", size=13, color="#1e1e1e")
    )
    
    # Add axis descriptions (sub-titles)
    fig.add_annotation(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text="<b>X축: 기술적 정렬도</b> - 입력한 아이디어와 특허의 개념적/원리적 일치 정도 (우측일수록 위험)",
        showarrow=False, font=dict(size=11, color="#555")
    )
    fig.add_annotation(
        x=-0.1, y=0.5, xref="paper", yref="paper", textangle=-90,
        text="<b>Y축: 분석 심도</b> - 특허 내용의 구체성 및 유사 데이터의 밀집도",
        showarrow=False, font=dict(size=11, color="#555")
    )
    
    # Add Quadrant Labels
    fig.add_annotation(x=0.85, y=0.9, text="<b>🚨 HIGH RISK</b>", showarrow=False, font=dict(color="#ff4b4b", size=15))
    fig.add_annotation(x=0.15, y=0.9, text="🔍 Reference", showarrow=False, font=dict(color="#7f8c8d", size=13))
    fig.add_annotation(x=0.85, y=0.1, text="💡 Potential Competitors", showarrow=False, font=dict(color="#6c5ce7", size=13))
    fig.add_annotation(x=0.15, y=0.1, text="📗 Distant Context", showarrow=False, font=dict(color="#28a745", size=13))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Revised Analysis Guide
    st.info("""
    🧭 **분석 가이드 및 축 설명**:
    - **X축 (기술적 정렬도)**: 아이디어의 핵심 기술 사상이 검색된 특허와 얼마나 일치하는지 나타냅니다. 1.0에 가까울수록 직설적인 모방이나 동일 기술일 확률이 높습니다.
    - **Y축 (분석 심도)**: 해당 특허가 다루는 기술의 범위와 복잡도, 그리고 우리 엔진의 유사 판단 근거가 얼마나 탄탄한지를 나타냅니다.
    
    **4분면 해석**:
    1. **우측 상단 (🚨 HIGH RISK)**: 기술 원리가 거의 일치하며 내용도 구체적인 **핵심 위험** 영역입니다.
    2. **우측 하단 (💡 Potential Competitors)**: 원리는 유사하나 표현이나 기술 수준이 다른 **잠재적 경쟁** 영역입니다. 회피 설계 검토가 필요합니다.
    3. **좌측 상단 (🔍 Reference)**: 일부 키워드나 구성은 겹치나 기술적 사상이 다른 **단순 참고** 영역입니다.
    4. **좌측 하단 (📗 Distant Context)**: 관련성은 낮지만 기술 분야가 겹칠 수 있는 **단순 배경 기술**입니다.
    """)
 