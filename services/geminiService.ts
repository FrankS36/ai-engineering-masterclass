import { GoogleGenAI } from "@google/genai";
import { Chapter } from "../types";

// Initialize Gemini Client
// IMPORTANT: Expects process.env.API_KEY to be available.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const askTutor = async (
  question: string,
  chapter: Chapter,
  history: { role: 'user' | 'model'; text: string }[]
): Promise<string> => {
  try {
    const model = 'gemini-2.5-flash';
    
    // Construct context from the current chapter content
    const systemInstruction = `
      You are an expert AI Engineering Tutor. 
      The user is studying a specific chapter from an "AI Engineering" book.
      
      Here is the content of the current chapter they are reading:
      ---
      Title: ${chapter.title}
      ${chapter.content}
      ---
      
      Answer the user's question based PRIMARILY on the provided chapter content.
      If the answer is not in the chapter, you may use your general knowledge but mention that it covers topics outside the current notes.
      Be concise, encouraging, and academic in tone. Use Markdown for formatting.
    `;

    const chat = ai.chats.create({
      model: model,
      config: {
        systemInstruction: systemInstruction,
      },
      history: history.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.text }]
      }))
    });

    const response = await chat.sendMessage({ message: question });
    
    return response.text || "I apologize, I couldn't generate a response.";

  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Sorry, I encountered an error while trying to reach the AI tutor. Please check your API configuration.";
  }
};

export const generateImage = async (prompt: string): Promise<string | null> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: [{ text: prompt }],
      },
      config: {
        // No specific image config needed for flash-image default
      }
    });

    if (response.candidates?.[0]?.content?.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData && part.inlineData.data) {
          return `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
        }
      }
    }
    return null;
  } catch (error) {
    console.error("Gemini Image Gen Error:", error);
    return null;
  }
};

export interface SearchResult {
  text: string;
  sources: { title: string; uri: string }[];
}

export const getLiveUpdates = async (topic: string): Promise<SearchResult> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `Find the absolute latest news, releases, and technical breakthroughs regarding "${topic}" from the last 3 months. Summarize the top 3 developments in a bulleted list.`,
      config: {
        tools: [{ googleSearch: {} }],
      },
    });

    const text = response.text || "No updates found.";
    
    // Extract grounding chunks for citations
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const sources = chunks
      .filter((c: any) => c.web?.uri && c.web?.title)
      .map((c: any) => ({
        title: c.web.title,
        uri: c.web.uri
      }));

    // Deduplicate sources based on URI
    const uniqueSources = Array.from(new Map(sources.map((item:any) => [item.uri, item])).values()) as { title: string; uri: string }[];

    return { text, sources: uniqueSources };
  } catch (error) {
    console.error("Gemini Search Error:", error);
    return { text: "Unable to fetch live updates at this moment.", sources: [] };
  }
};

export const runQuickPrompt = async (prompt: string): Promise<string> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: {
                maxOutputTokens: 150, // Keep it short for the sandbox
            }
        });
        return response.text || "No response generated.";
    } catch (error) {
        return "Error generating response.";
    }
}
