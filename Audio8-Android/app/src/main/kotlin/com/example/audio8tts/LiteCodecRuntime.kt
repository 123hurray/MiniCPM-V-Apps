package com.example.audio8tts
import android.content.Context
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.util.concurrent.CancellationException
import java.util.function.BooleanSupplier

/** Standalone pinned runtime, CPU INT8. No GPU/NPU delegate is selected implicitly. */
class LiteCodecRuntime(private val context: Context, private val mode: String) : AutoCloseable {
    private var model: CompiledModel?=null
    private var input: TensorBuffer?=null
    private var output: TensorBuffer?=null
    private var activeThreads=0
    private fun ensure(threads: Int) {
        if(model!=null && activeThreads==threads)return
        close()
        if(!(if(mode=="cpu_int8")LiteCodecStore.ready(context) else LiteFloatStore.ready(context)))throw IllegalStateException("请先下载 LiteRT INT8 codec")
        val start=SystemClock.elapsedRealtime()
        val options=if(mode=="gpu_fp16")CompiledModel.Options(Accelerator.GPU) else CompiledModel.Options(Accelerator.CPU)
        options.cpuOptions=CompiledModel.CpuOptions(threads.coerceIn(1,8),null,if(mode=="cpu_int8")java.io.File(context.cacheDir,"codec-int8-56fcb49-t$threads.cache").absolutePath else null)
        try {
            val m=CompiledModel.create(java.io.File(if(mode=="cpu_int8")LiteCodecStore.dir(context) else LiteFloatStore.dir(context),if(mode=="cpu_int8")"codec_decoder_int8_T128.tflite" else "codec_decoder_fp16_T128.tflite").absolutePath,options,null)
            model=m;input=m.createInputBuffer("codes","decode");output=m.createOutputBuffer("wav","decode");activeThreads=threads
        }catch(e:Throwable){close();throw e}
        ModelStore.log(context,"codec_backend=LiteRT_$mode init_ms=${SystemClock.elapsedRealtime()-start} threads=$threads")
    }
    fun decode(codes: IntArray,frames: Int,threads: Int,windows: Boolean,cancel: BooleanSupplier): FloatArray {
        CodecContract.validate(codes,frames)
        if(frames>128&&!windows)throw IllegalArgumentException("语音超过 128 帧；请显式开启上下文窗口解码（长语音音质需验证）")
        val total=SystemClock.elapsedRealtime();ensure(threads)
        val result=FloatArray(frames*2048);var done=0;var calls=0
        while(done<frames){
            if(cancel.asBoolean)throw CancellationException("codec cancelled")
            val start=if(done==0)0 else maxOf(0,done-64)
            val count=minOf(128,frames-start)
            val t=SystemClock.elapsedRealtime()
            input!!.writeInt(CodecContract.window(codes,frames,start))
            model!!.run(mapOf("codes" to input!!),mapOf("wav" to output!!),"decode")
            val pcm=output!!.readFloat()
            require(pcm.size==128*2048){"LiteRT codec 输出形状错误"}
            System.arraycopy(pcm,(done-start)*2048,result,done*2048,(start+count-done)*2048)
            done=start+count;calls++
            ModelStore.log(context,"codec_backend=LiteRT_$mode call=$calls start=$start valid_frames=$count run_copy_ms=${SystemClock.elapsedRealtime()-t}")
        }
        val rms=CodecContract.rms(result)
        if(rms<1e-7||rms>2.0)throw IllegalStateException("codec PCM 静音或幅值异常 rms=$rms")
        if(cancel.asBoolean)throw CancellationException("codec cancelled")
        ModelStore.log(context,"codec_backend=LiteRT_$mode total_ms=${SystemClock.elapsedRealtime()-total} frames=$frames calls=$calls rms=$rms windowed=${frames>128}")
        return result
    }
    override fun close(){output?.close();output=null;input?.close();input=null;model?.close();model=null;activeThreads=0}
}
