import pickle

whisper_path = '/home/rock/project/conv-emotion/data/wisper/iemocap/iemocap_whisper_large_v3.pkl'

try:
    with open(whisper_path, 'rb') as f:
        data = pickle.load(f)
        print("✅ 读取成功！")
        print(f"📊 总特征数: {len(data)}")
        
        # 打印前 5 个 Key，看看它是 'Ses01...' 还是带路径的
        print("\n🔍 Key 命名示例 (前5个):")
        keys = list(data.keys())
        for k in keys[:5]:
            print(f"  - {k}")
            
except Exception as e:
    print(f"❌ 读取仍然失败: {e}")
    print("💡 提示: 如果还是 MARK 错误，请尝试在命令行运行 'file " + whisper_path + "' 看看它到底是不是数据文件。")