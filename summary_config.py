#!/usr/bin/env python3
"""
Summary Configuration - Customize your AI summarization here!

This file contains all the settings and prompts for how you want your 
transcriptions to be summarized.
"""

class SummaryConfig:
    """Configuration class for AI summarization"""
    
    # ==========================================
    # SUMMARY STYLE CONFIGURATION
    # ==========================================
    
    # Choose your summary style (uncomment one):
    STYLE = "meeting_focused"  # Focus on meetings and business discussions
    # STYLE = "action_items"   # Focus on tasks and action items  
    # STYLE = "key_points"     # Focus on main topics and key points
    # STYLE = "detailed"       # More detailed summaries
    # STYLE = "bullet_points"  # Bullet point format
    # STYLE = "custom"         # Use your custom template below
    
    # ==========================================
    # CUSTOM PROMPT TEMPLATES
    # ==========================================
    
    PROMPTS = {
        "meeting_focused": {
            "prefix": "Create a concise summary of this conversation emphasizing key outcomes: ",
            "template": """
            Summary Instructions:
            - Focus on main conclusions reached
            - Highlight important action items with owners
            - Include specific numbers, dates, and deadlines
            - Note any follow-up meetings or next steps
            - Keep it conversational and clear
            
            Text: {text}
            
            Summary:"""
        },
        
        "action_items": {
            "prefix": "Extract action items, tasks, and decisions from this conversation: ",
            "template": """
            Focus on:
            • Action items and tasks assigned
            • Deadlines and timelines
            • Decisions made
            • People responsible
            
            Conversation: {text}
            
            Action Items:"""
        },
        
        "key_points": {
            "prefix": "Identify the main topics and key points from this discussion: ",
            "template": """
            Extract:
            - Main topics discussed
            - Important facts or figures
            - Key insights or conclusions
            - Notable quotes or statements
            
            Text: {text}
            
            Key Points:"""
        },
        
        "detailed": {
            "prefix": "Provide a comprehensive summary covering all important aspects: ",
            "template": """
            Create a detailed summary including:
            - Context and background
            - Main discussion points
            - Conclusions or outcomes
            - Next steps if mentioned
            
            Content: {text}
            
            Detailed Summary:"""
        },
        
        "bullet_points": {
            "prefix": "Convert this conversation into bullet points highlighting main topics: ",
            "template": """
            Format as bullet points:
            • Main topic 1
            • Main topic 2
            • Key decisions
            • Important details
            
            Text: {text}
            
            Summary:
            •"""
        },
        
        "custom": {
            "prefix": "Analyze this conversation and provide insights: ",
            "template": """
            Please analyze this conversation and provide:
            
            🎯 KEY TAKEAWAYS:
            - Most important points discussed
            - Critical decisions or agreements
            
            📋 ACTION ITEMS:
            - Who needs to do what
            - When tasks are due
            
            🔍 INSIGHTS:
            - Patterns or trends mentioned
            - Potential concerns or opportunities
            
            ⏰ FOLLOW-UP:
            - Next steps or meetings needed
            
            Text: {text}
            
            Analysis:"""
        }
    }
    
    # ==========================================
    # ADVANCED SETTINGS
    # ==========================================
    
    # Summary length settings (percentage of original text)
    TARGET_COMPRESSION = 0.4  # 40% of original length
    MIN_COMPRESSION = 0.15     # Minimum 15% of original
    
    # Quality settings
    INCLUDE_TIMESTAMPS = False   # Add time references to summaries
    INCLUDE_CONFIDENCE = False   # Add confidence scores
    FILTER_LOW_CONFIDENCE = True # Skip very low confidence transcriptions
    MIN_CONFIDENCE_THRESHOLD = -1.0  # Minimum confidence to include
    
    # Content filtering
    SKIP_VERY_SHORT = True      # Skip transcriptions under N words
    MIN_WORDS_THRESHOLD = 5     # Minimum words to process
    
    # ==========================================
    # CUSTOM FUNCTIONS (ADVANCED)
    # ==========================================
    
    @staticmethod
    def custom_preprocessor(text):
        """
        Add your custom text preprocessing here
        
        Args:
            text (str): Original transcription text
            
        Returns:
            str: Processed text ready for summarization
        """
        # Example: Remove filler words
        filler_words = ["um", "uh", "like", "you know"]
        words = text.split()
        cleaned_words = [w for w in words if w.lower() not in filler_words]
        
        # Example: Add context if needed
        # if "meeting" not in text.lower():
        #     text = "In this discussion: " + text
        
        return " ".join(cleaned_words)
    
    @staticmethod
    def custom_postprocessor(summary):
        """
        Add your custom summary post-processing here
        
        Args:
            summary (str): AI-generated summary
            
        Returns:
            str: Final processed summary
        """
        # Example: Ensure proper capitalization
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Example: Add standard footer
        # summary += " [AI Generated Summary]"
        
        return summary

# ==========================================
# EASY CUSTOMIZATION EXAMPLES
# ==========================================

"""
EXAMPLE CUSTOMIZATIONS:

1. BUSINESS MEETING FOCUS:
   STYLE = "meeting_focused"
   
2. PERSONAL CONVERSATION STYLE:
   STYLE = "key_points" 
   
3. TASK TRACKING:
   STYLE = "action_items"

4. CUSTOM STYLE - Edit the "custom" template above with:
   STYLE = "custom"
   Then modify PROMPTS["custom"]["template"] with your requirements

5. DETAILED CONTROL - Modify any of the advanced settings above
"""
