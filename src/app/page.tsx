"use client";

import { useState, useRef, useEffect } from "react";
import { Bot, Upload, Database, Trash2, FileText, Send, CheckCircle2, AlertCircle, X } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const SUGGESTED_TAGS = [
  "推荐一款降噪耳机",
  "人体工学椅 V2 多少钱？",
  "七天无理由退货的条件是什么？",
  "数码产品保修期多久？",
];

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error', message: string } | null>(null);
  const [isResetting, setIsResetting] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [kbDocuments, setKbDocuments] = useState<any[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 处理文件选择
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setUploadStatus(null);
      
      // 自动触发上传
      await uploadFile(selectedFile);
      
      // 清空 input，以便可以重复选择同一文件
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // 调用 upload 接口
  const uploadFile = async (selectedFile: File) => {
    if (!selectedFile) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setUploadStatus({ type: 'success', message: `上传成功！已处理 ${data.processedChunks} 个块。` });
      } else {
        setUploadStatus({ type: 'error', message: `上传失败: ${data.error}` });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus({ type: 'error', message: "上传发生错误" });
    } finally {
      setIsUploading(false);
    }
  };

  // 调用 documents 接口查看知识库
  const handleViewKnowledgeBase = async () => {
    try {
      const response = await fetch("/api/documents?page=1&pageSize=10");
      const data = await response.json();
      if (response.ok) {
        setKbDocuments(data.documents || []);
        setIsDialogOpen(true);
      } else {
        alert("获取知识库失败: " + data.error);
      }
    } catch (error) {
      console.error("Fetch documents error:", error);
      alert("获取知识库发生错误");
    }
  };

  // 调用 reset 接口
  const handleReset = async () => {
    if (!confirm("确定要重置知识库吗？此操作不可恢复。")) return;
    
    setIsResetting(true);
    try {
      const response = await fetch("/api/reset", {
        method: "POST",
      });
      const data = await response.json();
      if (response.ok) {
        alert("知识库已成功重置！");
        setFile(null);
        setUploadStatus(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      } else {
        alert("重置失败: " + data.error);
      }
    } catch (error) {
      console.error("Reset error:", error);
      alert("重置发生错误");
    } finally {
      setIsResetting(false);
    }
  };

  // 发送聊天消息
  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    // 先把用户的新消息放到界面上
    const newMessages: Message[] = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setIsLoading(true);
    setInputValue("");

    try {
      // 调用 chat 接口，连同历史记录一起发送
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: text,
          history: messages // 把之前的历史记录传给后端
        }),
      });
      
      if (response.status === 501) {
        // Mock response if backend is not implemented yet
        setTimeout(() => {
          setMessages(prev => [
            ...prev, 
            { role: "assistant", content: "抱歉，聊天接口尚未完全实现，我是模拟的回复。您说的是：" + text }
          ]);
          setIsLoading(false);
        }, 1000);
        return;
      }

      const data = await response.json();
      if (response.ok) {
        setMessages(prev => [...prev, { role: "assistant", content: data.answer || data.reply || data.message }]);
      } else {
        setMessages(prev => [...prev, { role: "assistant", content: "请求出错：" + (data.error || "未知错误") }]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { role: "assistant", content: "网络错误，请稍后再试。" }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = () => {
    if (inputValue.trim()) {
      handleSendMessage(inputValue.trim());
    }
  };

  const handleTagClick = (tag: string) => {
    handleSendMessage(tag);
  };

  return (
    <div className="flex flex-col h-screen w-full bg-gray-50 text-gray-900 font-sans">
      {/* 顶部 Header */}
      <header className="flex items-center gap-2 px-6 py-4 bg-white border-b border-gray-200 shrink-0">
        <Bot className="w-6 h-6 text-blue-600" />
        <h1 className="text-xl font-bold text-gray-800">智能客服</h1>
      </header>

      {/* 主体内容区 */}
      <main className="flex-1 flex overflow-hidden p-6 gap-6">
        {/* 左侧控制面板 */}
        <aside className="w-80 flex flex-col gap-6 shrink-0">
          {/* 上传区域 Card */}
          <div className="bg-white rounded-xl shadow-sm p-6 flex flex-col gap-5 border border-gray-100">
            <div className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-gray-800" />
              <h2 className="font-bold text-gray-800">Upload Profile</h2>
            </div>
            
            <div className="flex items-center gap-3">
              <label className={`cursor-pointer ${isUploading ? 'bg-blue-100 text-blue-400' : 'bg-blue-50 text-blue-600 hover:bg-blue-100'} px-4 py-2 rounded-full text-sm font-medium transition-colors`}>
                {isUploading ? "Uploading..." : "Choose File"}
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                  disabled={isUploading}
                  accept=".txt,.pdf,.md,.csv" // 根据实际情况调整
                />
              </label>
              <span className="text-sm text-gray-400 truncate max-w-[120px]" title={file?.name || "No file chosen"}>
                {file ? file.name : "No file chosen"}
              </span>
            </div>
            
            {uploadStatus && (
              <div className={`flex items-start gap-2 text-sm p-3 rounded-md ${uploadStatus.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                {uploadStatus.type === 'success' ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" /> : <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
                <p className="flex-1 break-words">{uploadStatus.message}</p>
              </div>
            )}

            <button 
              onClick={handleViewKnowledgeBase} 
              className="mt-2 flex items-center justify-center gap-2 bg-gray-50 text-gray-600 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-100 transition-colors border border-gray-200"
            >
              <Database className="w-4 h-4" />
              View Knowledge Base
            </button>
          </div>

          {/* 危险区域 Card */}
          <div className="bg-white rounded-xl shadow-sm p-6 flex flex-col gap-5 border border-red-50">
            <div className="flex items-center gap-2">
              <Trash2 className="w-5 h-5 text-red-500" />
              <h2 className="font-bold text-red-600">Danger Zone</h2>
            </div>
            
            <button 
              onClick={handleReset} 
              disabled={isResetting}
              className={`w-full ${isResetting ? 'bg-red-100 text-red-300' : 'bg-red-50 text-red-600 hover:bg-red-100'} py-2.5 rounded-lg text-sm font-medium transition-colors`}
            >
              {isResetting ? "Resetting..." : "Reset Knowledge Base"}
            </button>
          </div>
        </aside>

        {/* 右侧聊天区 */}
        <section className="flex-1 bg-white rounded-xl shadow-sm flex flex-col overflow-hidden border border-gray-100">
          {/* 聊天记录 */}
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center">
                <FileText className="w-12 h-12 text-gray-200 mb-4" />
                <p className="text-gray-600 mb-2 font-medium">👋 您好，我是智能客服。请问有什么可以帮您？</p>
                <p className="text-gray-400 text-sm">您可以问我：这款耳机有降噪吗？怎么退货？</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed ${msg.role === 'user' ? 'bg-blue-500 text-white rounded-br-sm' : 'bg-gray-100 text-gray-800 rounded-bl-sm'}`}>
                    {msg.content}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-500 rounded-2xl rounded-bl-sm px-5 py-3 text-sm">
                  <span className="animate-pulse">思考中...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* 底部输入区 */}
          <div className="p-4 bg-white border-t border-gray-100">
            {/* 快捷输入 Tags */}
            <div className="flex flex-wrap gap-2 mb-4">
              {SUGGESTED_TAGS.map(tag => (
                <button 
                  key={tag} 
                  onClick={() => handleTagClick(tag)} 
                  disabled={isLoading}
                  className="bg-gray-50 text-gray-600 px-3 py-1.5 rounded-full text-xs font-medium hover:bg-gray-100 hover:text-gray-900 transition-colors border border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {tag}
                </button>
              ))}
            </div>
            
            {/* 输入框 */}
            <div className="flex items-center gap-2 border border-gray-300 rounded-full px-4 py-2 bg-white focus-within:ring-2 focus-within:ring-blue-100 focus-within:border-blue-400 transition-all">
              <input 
                type="text" 
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                placeholder="Ask a question about the AI Shop..." 
                className="flex-1 outline-none text-sm bg-transparent"
                onKeyDown={e => e.key === 'Enter' && handleSend()}
              />
              <button 
                onClick={handleSend} 
                disabled={!inputValue.trim() || isLoading}
                className={`p-2 rounded-full transition-colors flex items-center justify-center ${!inputValue.trim() || isLoading ? 'bg-gray-100 text-gray-400' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* 知识库详情弹窗 */}
      {isDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-white rounded-xl shadow-lg w-full max-w-3xl max-h-[80vh] flex flex-col overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b border-gray-100">
              <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-600" />
                知识库详细数据
              </h2>
              <button 
                onClick={() => setIsDialogOpen(false)}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              {kbDocuments.length === 0 ? (
                <div className="text-center text-gray-500 py-10">知识库中暂无数据。</div>
              ) : (
                <div className="flex flex-col gap-4">
                  {kbDocuments.map((doc, idx) => (
                    <div key={idx} className="border border-gray-200 rounded-lg p-4 bg-gray-50/50">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="px-2.5 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-md">
                          Chunk {idx + 1}
                        </span>
                        <span className="text-xs text-gray-500 truncate" title={doc.metadata?.source}>
                          {doc.metadata?.source || "未知来源"}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                        {doc.pageContent}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="p-4 border-t border-gray-100 bg-gray-50 text-right text-xs text-gray-500">
              共加载 {kbDocuments.length} 条数据
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
