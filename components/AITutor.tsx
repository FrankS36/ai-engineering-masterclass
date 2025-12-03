import React, { useState, useRef, useEffect } from 'react';
import { Chapter, Message } from '../types';
import { askTutor } from '../services/geminiService';
import { Send, Bot, User, Loader2 } from 'lucide-react';

interface AITutorProps {
  chapter: Chapter;
}

export const AITutor: React.FC<AITutorProps> = ({ chapter }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', text: `Hello! I'm your AI Tutor. I've read **${chapter.title}**. What would you like to know?` }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Reset chat when chapter changes
  useEffect(() => {
    setMessages([
      { role: 'model', text: `Hello! I'm your AI Tutor. I've read **${chapter.title}**. What would you like to know?` }
    ]);
  }, [chapter.id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const responseText = await askTutor(input, chapter, messages);
      setMessages(prev => [...prev, { role: 'model', text: responseText }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'model', text: "Sorry, something went wrong.", isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to render markdown-ish text in chat bubbles
  const renderMessageText = (text: string) => {
    // Basic bold formatting
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return part;
    });
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] bg-white border border-stone-200 rounded-xl shadow-sm overflow-hidden m-8 max-w-4xl mx-auto">
      <div className="bg-stone-900 p-4 text-white flex items-center gap-3">
        <div className="bg-white/10 p-2 rounded-full text-brand-500">
            <Bot size={20} />
        </div>
        <div>
            <h3 className="font-semibold text-white">AI Tutor</h3>
            <p className="text-stone-400 text-xs">Ask me anything about this chapter</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-stone-50">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                msg.role === 'user' ? 'bg-stone-200 text-stone-600' : 'bg-brand-100 text-brand-600'
              }`}>
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                msg.role === 'user' 
                  ? 'bg-stone-800 text-white rounded-tr-none' 
                  : 'bg-white border border-stone-200 text-stone-800 rounded-tl-none shadow-sm'
              }`}>
                {msg.isError ? <span className="text-red-500">{msg.text}</span> : <div className="whitespace-pre-wrap">{renderMessageText(msg.text)}</div>}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[80%]">
              <div className="w-8 h-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center shrink-0">
                <Bot size={16} />
              </div>
              <div className="bg-white border border-stone-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2 text-stone-500 text-sm">
                <Loader2 size={16} className="animate-spin" />
                Thinking...
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-white border-t border-stone-200">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the notes..."
            className="flex-1 border border-stone-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent text-sm bg-stone-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-brand-600 hover:bg-brand-700 text-white p-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
};