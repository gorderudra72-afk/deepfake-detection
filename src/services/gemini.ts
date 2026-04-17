import { GoogleGenAI, Type } from "@google/genai";

let aiInstance: GoogleGenAI | null = null;

function getAiClient() {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is missing. Please configure it in the Secrets panel.");
    }
    aiInstance = new GoogleGenAI({ apiKey });
  }
  return aiInstance;
}

export interface AnalysisResult {
  score: number; // 0-100, where 100 is highly likely to be a deepfake
  verdict: 'Authentic' | 'Suspicious' | 'High Risk';
  isReal: boolean;
  confidence: number;
  findings: string[];
  explanation: string;
}

export async function analyzeMedia(base64Data: string, mimeType: string): Promise<AnalysisResult> {
  const isVideo = mimeType.startsWith('video/');
  const prompt = `
    You are a world-class digital forensics expert specializing in deepfake and synthetic media detection.
    Analyze the provided ${isVideo ? 'video' : 'image'} for any signs of digital manipulation, AI generation, or deepfake artifacts.
    
    Look specifically for facial-centric biometrics and artifacts:
    1. Facial inconsistencies: asymmetric pupils, unnatural eye-glint, non-matching iris patterns, or "liquid" teeth.
    2. Boundary artifacts: fine-line "halos" or blurring where the face meets the hair or neck.
    3. Facial biometrics: unnatural blinking patterns (or lack thereof), inconsistent facial micro-expressions, and skin porosity that looks smoothed or plastic.
    4. Lighting & Shadows: facial shadows that don't match the primary light source in the background.
    5. Morphing: visible jitters or "shimmering" during heavy facial movement or occlusion (e.g., a hand passing in front of a face).
    
    If multiple faces are present, analyze the primary subject most thoroughly.
    ${isVideo ? '6. Temporal artifacts (flickering, face-swapping jitters, motion blurring errors).\n7. Audio-visual sync anomalies.' : ''}
    
    Final Goal: Determine if this media is REAL (Authentic) or FAKE (Deepfake/Manipulated).
    Respond ONLY with a valid JSON object.
  `;

  try {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType,
                data: base64Data.split(',')[1] || base64Data
              }
            }
          ]
        }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: {
              type: Type.NUMBER,
              description: "Deepfake probability score from 0 to 100."
            },
            verdict: {
              type: Type.STRING,
              enum: ["Authentic", "Suspicious", "High Risk"],
              description: "The summary verdict of the analysis."
            },
            isReal: {
              type: Type.BOOLEAN,
              description: "True if specifically determined to be authentic/real, False if likely fake/manipulated."
            },
            confidence: {
              type: Type.NUMBER,
              description: "The AI's confidence in this verdict (0-100)."
            },
            findings: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Specific detected artifacts or suspicious points."
            },
            explanation: {
              type: Type.STRING,
              description: "A detailed breakdown of the forensics analysis."
            }
          },
          required: ["score", "verdict", "isReal", "findings", "explanation", "confidence"]
        }
      }
    });

    const text = response.text || '';
    // Extract the JSON block more robustly
    let jsonStr = text;
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      jsonStr = jsonMatch[0];
    }
    
    if (!jsonStr.trim()) {
      throw new Error("The AI provided an empty analysis. This can happen with very large or complex files. Please try a smaller image/video.");
    }
    
    try {
      const result = JSON.parse(jsonStr);
      return result as AnalysisResult;
    } catch (parseError) {
      console.error("JSON Parse Error. Raw text:", text);
      throw new Error("The forensics engine returned an unreadable report format. Please try again.");
    }
  } catch (error: any) {
    console.error("Analysis failed:", error);
    throw new Error(error.message || "Failed to analyze media. Please try again.");
  }
}
