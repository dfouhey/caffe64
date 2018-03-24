.intel_syntax noprefix
.globl _start
/*
   ______        ____ ____         _____  __ __
  / ____/____ _ / __// __/___     / ___/ / // /
 / /    / __ `// /_ / /_ / _ \   / __ \ / // /_
/ /___ / /_/ // __// __//  __/  / /_/ //__  __/
\____/ \__,_//_/  /_/   \___/   \____/   /_/
    
(c) 2018 The Caffe64 Development Team
*/


/* 
exec train network.txt weights.bin optimizer.txt N inputX.txt inputY.txt
exec test network.txt weights.bin N inputX.txt outputY.txt

Net format:
------------
#layers
layer_spec_1
...
layer_spec_N

Optim format:
------------
lr lr_step_epoch lr_step_factor weight_decay momentum_gamma numepochs
*/


/* common definitions */

/* system calls */
.equ SYS_READ,      0
.equ SYS_WRITE,     1
.equ SYS_OPEN,      2
.equ SYS_CLOSE,     3
.equ SYS_LSEEK,     8
.equ SYS_BRK,       12
.equ SYS_ACCESS,    21
.equ SYS_EXIT,      60
.equ SYS_TIME,      201

.equ SYS_STDERR,    0
.equ SYS_STDOUT,    1

/* 
Net format (all quads):
0: Num Layers = L
1 - L: pointer to layer struct
L+1: diagnostic (low dword is a float set by loss functions) 

Layer struct format (all quads):
0: layer type id  (by stub)
1: back-pointer to the network (i.e., count, layer *) (by stub)
2: layer num (by stub)
3: N 
4: F (in) 
5: K (out)
6: P # parameters
7: pointer to output data (NxK)
8: pointer to bottom diff (NxF)
9: pointer to parameters (P)
10: pointer to diff parameters (P)
11: pointer to momentum for parameters (P)
*/

.equ LAYER_OFF_ID, 0
.equ LAYER_OFF_BP, 8
.equ LAYER_OFF_NUM, 16
.equ LAYER_OFF_N, 24
.equ LAYER_OFF_F, 32
.equ LAYER_OFF_K, 40
.equ LAYER_OFF_P, 48
.equ LAYER_OFF_DAT, 56
.equ LAYER_OFF_DIFF, 64
.equ LAYER_OFF_PDAT, 72
.equ LAYER_OFF_PDIFF, 80
.equ LAYER_OFF_PMOM, 88

/* layers */
.equ LAYER_COUNT, 7
.equ LAYER_SIZE, 96

/* layer ids */
.equ LAYER_INPUT,   0
.equ LAYER_LINEAR,  1
.equ LAYER_RELU,    2
.equ LAYER_TANH,    3
.equ LAYER_SOFTMAX, 4
.equ LAYER_SCE,     5
.equ LAYER_L2,      6

/* layer names */
.equ LNAME_INPUT,   0x00706e69
.equ LNAME_LINEAR,  0x006e696c
.equ LNAME_RELU,    0x006c6572
.equ LNAME_TANH,    0x00686e74
.equ LNAME_SOFTMAX, 0x00786d73
.equ LNAME_SCE,     0x00656373
.equ LNAME_L2,      0x006c326c

/* macros */

/* negate the xmm register in xmmr, using the r32 register in r32r */
.macro negatexmm xmmr=xmm0 r32r=eax
    movd \r32r, \xmmr
    btc \r32r, 31
    movd \xmmr, \r32r
.endm

.equ INIT_HEAP_SIZE, 4096
.data


helpStr: .asciz "caffe64 train network.txt weights.bin optimizer.txt N inputX.txt inputY.txt\n  or\ncaffe64 test network.txt weights.bin N inputX.txt outputY.txt\n"
.equ helpStrLen, 143

couldntFindStr: .asciz "Couldn't find file "
.equ couldntFindStrLen, 19

/* status updates */
loadingXStr: .asciz "Loading X..."
.equ loadingXStrLen, 12
loadingYStr: .asciz "Loading Y..."
.equ loadingYStrLen, 12
parsingNetStr: .asciz "Parsing net..."
.equ parsingNetStrLen, 14
loadingNetStr: .asciz "Loading net..."
.equ loadingNetStrLen, 14
lrDropStr: .asciz "LR Drop. Multiplying by "
.equ lrDropStrLen, 24

statStringTr1: .asciz "[Training] Epoch "
.equ statStringTr1Len, 17
statStringTe1: .asciz "[Testing] Epoch "
.equ statStringTe1Len, 16
statStringTr2: .asciz " MBIter "
.equ statStringTr2Len, 8
statStringTr3: .asciz " LR "
.equ statStringTr3Len, 4
statStringTr4: .asciz " Loss "
.equ statStringTr4Len, 6
endingString: .asciz "Finished normally in "
.equ endingStringLen, 21

/* local macro that writes a string to stdout */
.macro emitString strn="" writenl=0
    mov rax, SYS_WRITE
    mov rdi, SYS_STDOUT    
    lea rsi, [\strn]
    mov rdx, \strn\()Len
    syscall
    .if \writenl
        mov rdi, SYS_STDOUT
        call printnl
    .endif
.endm

/* dat global var */
netPtr: .quad 0
shuffleBufferPtr: .quad 0
currentN: .quad 0
currentF: .quad 0
currentK: .quad 0
currentPtrX: .quad 0
currentPtry: .quad 0
currentBuffPtry: .quad 0
currentMinibatchSize: .quad 0
currentMinibatchesPerEpoch: .quad 0
currentMinibatchesRemainder: .quad 0

baseLR: .float 0
epochDecay: .quad 0
decayLR: .float 0
weightDecay: .float 0
momentumGamma: .float 0
numEpochs: .quad 1
currentLR: .float 0

/* status stuff */
printPhase: .quad 0
currentEpochs: .quad 0
currentIter: .quad 0
currentLoss: .float 0
.equ iterPrintRate, 20

clockStart: .quad 0
clockEnd:.quad 0

.equ RAND_BUFF_SIZE, 128
devRandom: .asciz "/dev/urandom"
byteReserve: .fill RAND_BUFF_SIZE, 1, 0
randBufferPtr: .quad RAND_BUFF_SIZE 


bannerString: .asciz "   ______        ____ ____         _____  __ __\n  / ____/____ _ / __// __/___     / ___/ / // /\n / /    / __ `// /_ / /_ / _ \\   / __ \\ / // /_\n/ /___ / /_/ // __// __//  __/  / /_/ //__  __/\n\\____/ \\__,_//_/  /_/   \\___/   \\____/   /_/ \n"
.equ bannerStringLen, 238

chrtype: .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

NaNStr: .asciz "NaN"
InfStr: .asciz "Inf"


layerId:
.quad LAYER_INPUT, LAYER_LINEAR, LAYER_RELU, LAYER_TANH, LAYER_SOFTMAX, LAYER_SCE, LAYER_L2

layerName:
.quad LNAME_INPUT, LNAME_LINEAR, LNAME_RELU, LNAME_TANH, LNAME_SOFTMAX, LNAME_SCE, LNAME_L2

/* an init takes rdi: ptr to layer, rsi: spec string, returns nothing */ 
layerInitTable:
.quad inputInit, linearInit, actInit, actInit, actInit, actInit, actInit 

/* a forward takes rdi: ptr to layer, and maps prev layer's output to its output */ 
layerForwardTable:
.quad defaultForward, linearForward, reluForward, tanhForward, softmaxForward, sceForward, copyForward

/* a backward takes rdi: ptr to layer and sets deriv buffer. Loss layers also take rsi: ptr to label vector */
layerBackwardTable:
.quad defaultBackward, linearBackward, reluBackward, tanhBackward, defaultBackward, sceBackward, l2backward 


heapStart: .quad 0
heapSize: .quad 0
heapPtr: .quad 0


m_n2: .float -2.0
m_n1: .float -1.0
m_0: .float 0
m_0p1: .float 0.1
m_0p5: .float 0.5
m_1: .float 1.0
m_3: .float 3.0
m_10: .float 10.0
m_2oln2: .float 2.8853900817779268
/* yes, 2/ln(2) = 2*log2(e), but might as well just make another constant */
m_log2e: .float 1.4426950408889634

loadSizemismatchStr: .asciz "Layer doesn't fit, refusing to load\n"
.equ loadSizemismatchStrLen, 36

.text



helpExitErr:
    /* print help on stderr and exit */
    mov eax, SYS_WRITE
    mov rdi, SYS_STDERR
    lea rsi, [helpStr]
    mov rdx, helpStrLen
    syscall

    mov eax, SYS_EXIT
    mov rdi, 1 
    syscall

confirmExistsOrExit:
    /* confirm that a file named *rdi exists or exit */
    push r15
    mov r15, rdi

    mov eax, SYS_ACCESS
    /* rdi set before */
    xor rsi, rsi
    syscall
    cmp rax, 0
    jz 1f

    mov eax, SYS_WRITE
    mov rdi, SYS_STDERR
    lea rsi, [couldntFindStr]
    mov rdx, couldntFindStrLen
    syscall

    mov rdi, r15
    call strlen
    mov rdx, rax

    mov eax, SYS_WRITE
    mov rdi, SYS_STDERR
    mov rsi, r15
    mov rdx, rdx
    syscall

    mov rdi, SYS_STDERR
    call printnl

    mov eax, SYS_EXIT
    mov rdi, 1 
    syscall
    
1:  pop r15
    ret

loadInputXFile:
    /* given rdi ptr to input X file name, assumes currentN currentF set
      load X File and allocate the shufflebuffer */
    push r15
    mov r15, rdi

    mov rdi, [currentN]
    imul rdi, [currentF]
    shl rdi, 2
    call alloc
    mov [currentPtrX], rax

    mov rdi, r15
    mov rsi, [currentPtrX]
    mov rdx, [currentN]
    imul rdx, [currentF]
    call readmat

    mov rdi, qword ptr [currentN]
    shl rdi, 3
    call alloc
    mov [shuffleBufferPtr], rax

    pop r15
    ret

allocY:
    /* given currentN, currentF set
       alloc y (output / input) and shufflebuffer */
    mov rdi, qword ptr [currentN]
    shl rdi, 2
    call alloc
    mov [currentPtry], rax

    ret

loadInputYFile:
    /* given rdi ptr to input Y file name, assumes currentN, currentF */
    push r15
    mov r15, rdi

    call allocY

    mov rdi, r15
    mov rsi, [currentPtry]
    mov rdx, [currentN]
    call readmat

    pop r15
    ret

setupMinibatching:

    mov rdi, [netPtr]
    call getNetworkMinibatchSize
    mov [currentMinibatchSize], rax

    mov rax, [currentN]
    xor rdx, rdx
    mov rcx, [currentMinibatchSize]
    div rcx
    mov [currentMinibatchesPerEpoch], rax
    mov [currentMinibatchesRemainder], rdx

    mov rdi, qword ptr [currentMinibatchSize]
    shl rdi, 2
    call alloc
    mov [currentBuffPtry], rax

    ret


loadOptimizerSettings:
    /* given rdi ptr to char * */
    push rbp
    push rbx
    mov rbp, rsp
    sub rsp, 24

    lea rbx, [rbp-24]

    /* rdi set */
    mov rsi, rbx
    mov rdx, 6
    call readmat

    mov eax, [rbx]
    mov [baseLR], eax

    movss xmm0, [rbx+4]
    xor rax, rax
    cvtss2si eax, xmm0
    mov [epochDecay], eax

    mov eax, [rbx+8]
    mov [decayLR], eax

    mov eax, [rbx+12]
    mov [weightDecay], eax

    mov eax, [rbx+16]
    mov [momentumGamma], eax

    movss xmm0, [rbx+20]
    xor rax, rax
    cvtss2si eax, xmm0
    mov [numEpochs], rax

    add rsp, 24    
    pop rbx
    pop rbp
    ret

tic:
    mov rax, SYS_TIME
    lea rdi, [clockStart]
    syscall
    ret

toc:
    mov rax, SYS_TIME
    lea rdi, [clockEnd]
    syscall
    mov rax, [clockEnd]
    sub rax, [clockStart]
    ret


_start:
    call tic

    /* argc, argv */
    mov rdi, [rsp]
    lea rsi, [rsp+8]

    cmp rdi, 7
    jge 1f
    call helpExitErr
1:

    mov rax, [rsi+8]
    mov rax, [rax]
    shr rax, 8
    cmp al, 'r'
    jz 1f
    call mainTest
    jmp 3f
1:  call mainTrain

3:  
    
    emitString endingString
    call toc
    mov rdi, SYS_STDOUT
    mov rsi, rax
    call printInt
    mov rsi, 's'
    call printchar
    call printnl

    /* if we return signal no error */
    mov rax, SYS_EXIT
    xor rdi, rdi
    syscall


mainTrain:
    /*
    0       8     16          24          32            40  48         56
    caffe64 train network.txt weights.bin optimizer.txt N   inputX.txt inputY.txt
   
    r14 is argc to start, then epoch
    r15 is argv
    */
    mov r14, rdi
    mov r15, rsi

    mov rdi, SYS_STDOUT
    call banner

    /* check args */
    /* make sure we have enough args */
    cmp r14, 8
    jge 1f
    call helpExitErr
1:

    mov rdi, [r15+16]
    call confirmExistsOrExit

    mov rdi, [r15+32]
    call confirmExistsOrExit

    mov rdi, [r15+48]
    call confirmExistsOrExit

    mov rdi, [r15+56]
    call confirmExistsOrExit

    /* actually parse */

    emitString parsingNetStr 1

    mov rdi, [r15+16]
    call parseNet
    mov [netPtr], rax

    mov rdi, [r15+40]
    call sint
    mov [currentN], rax

    mov rdi, [netPtr]
    call getNetworkInputSize
    mov [currentF], rax

    mov rdi, [netPtr]
    call getNetworkOutputSize
    mov [currentK], rax

    emitString loadingXStr 1

    mov rdi, [r15+48]
    call loadInputXFile

    emitString loadingYStr 1

    mov rdi, [r15+56]
    call loadInputYFile

    call setupMinibatching

    mov rdi, [r15+32]
    call loadOptimizerSettings

    mov rdi, [netPtr]
    call netOptimInit

    /* train loop */
    movss xmm0, [baseLR]
    movss [currentLR], xmm0

    xor r14, r14
1:  mov [currentEpochs], r14
    mov rdi, [netPtr]
    call runEpoch
    add r14, 1

    mov rax, r14
    xor rdx, rdx
    idiv rax, [epochDecay]
    cmp rdx, 0
    jnz 2f
    /* update lr */
    movss xmm0, [currentLR]
    mulss xmm0, [decayLR]
    movss [currentLR], xmm0

    emitString lrDropStr
    movss xmm0, [decayLR]
    mov rdi, SYS_STDOUT
    call printFloat
    call printnl


2:    
    cmp r14, [numEpochs]
    jnz 1b

    /* /train loop */

    mov rdi, [netPtr]
    mov rsi, [r15+24]
    call saveNet

    ret


mainTest:
    /*
        0       8    16          24          32  40         48
        caffe64 test network.txt weights.bin N   inputX.txt outputY.txt

        r15 is argv
    */
    mov r15, rsi

    mov rdi, SYS_STDOUT
    call banner


    /* check args */
    mov rdi, [r15+16]
    call confirmExistsOrExit

    mov rdi, [r15+24]
    call confirmExistsOrExit

    mov rdi, [r15+40]
    call confirmExistsOrExit


    emitString parsingNetStr 1
    mov rdi, [r15+16]
    call parseNet
    mov [netPtr], rax

    emitString loadingNetStr 1
    mov rdi, [netPtr]
    mov rsi, [r15+24]
    call loadNet

    mov rdi, [r15+32]
    call sint
    mov [currentN], rax

    mov rdi, [netPtr]
    call getNetworkInputSize
    mov [currentF], rax

    mov rdi, [netPtr]
    call getNetworkOutputSize
    mov [currentK], rax

    call setupMinibatching

    emitString loadingXStr 1
    mov rdi, [r15+40]
    call loadInputXFile

    mov rdi, [r15+48]
    call runTest

    ret


runTest:
    /*  rdi -> ouptut name -> rbx
        r12 fid for output 
        r13 call-safe save how many to output
        r14 iterates up to r15   
        r15 num minibatches
    */
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbx, rdi

    mov rdi, [currentN]
    mov rsi, [shuffleBufferPtr]
    call identityperm

    /* figure out how many minibatch epochs to run -- might have
       to run a partial
     */
    mov r15, [currentMinibatchesPerEpoch]
    mov rax, [currentMinibatchesRemainder]
    cmp rax, 0
    jz 2f
    inc r15
2:

    /* set status flags so we can reuse */
    xor rax, rax
    mov [currentLoss], eax
    mov [currentLR], eax
    mov [currentEpochs], rax
    inc rax
    mov [numEpochs], rax
    mov [printPhase], rax

    /* setup file */
    mov eax, SYS_OPEN 
    mov rdi, rbx
    mov esi, 0x0641
    mov edx, 0777
    syscall
    mov r12, rax



    xor r14, r14    
runTest_epoch:

    mov [currentIter], r14
 
    /* ingest */
    mov rdi, [netPtr]
    mov rsi, [currentPtrX]
    mov rdx, [shuffleBufferPtr]
    mov rax, r14
    imul rax, [currentMinibatchSize]
    lea rdx, [rdx+8*rax]
    mov rcx, [currentMinibatchSize]

    /* how many to actually ingest */
    add rax, [currentMinibatchSize]
    cmp rax, [currentN]
    jl 2f
    sub rax, [currentN]
    sub rcx, rax 
2:  mov r13, rcx 
    call inputIngest

    mov rdi, [netPtr]
    call netForward

    /* deref to last layer */
    mov r8, [netPtr]
    mov rax, [r8]
    mov r8, [r8+8*rax]

    /* write on r12 r13 lines of K floats reading from r8's dat */
    mov rdi, r12
    mov rsi, r13
    mov rdx, [r8+LAYER_OFF_K]
    mov rcx, [r8+LAYER_OFF_DAT]
    call writemattofid

    call emitTestStatus

    add r14, 1
    cmp r14, r15
    jnz runTest_epoch


    /* close */
    mov eax, SYS_CLOSE
    mov rdi, r12
    syscall

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret


runEpoch:
    /* 
        r13 tracks the loss (float saved in call-safe r13)
        r15 iterates over minibatches 
     */
    push r13
    push r15

    /* shuffle */
    mov rdi, [currentN]
    mov rsi, [shuffleBufferPtr]
    call randperm

    /* also sets to zero */
    xor r13, r13

    xor r15, r15
runEpoch_mbl:
    mov [currentIter], r15

    /* figure out offset into shuffler */
    mov rax, r15
    imul rax, [currentMinibatchSize]

    /* ingest */
    mov rdi, [netPtr]
    mov rsi, [currentPtrX]
    mov rdx, [shuffleBufferPtr]
    lea rdx, [rdx+8*rax]
    mov rcx, [currentMinibatchSize]
    call inputIngest

    mov rax, r15
    imul rax, [currentMinibatchSize]

    mov rdi, [currentPtry]
    mov rsi, [currentBuffPtry]
    mov rdx, [shuffleBufferPtr]
    lea rdx, [rdx+8*rax]
    mov rcx, [currentMinibatchSize]
    call ingestVector

    mov rdi, [netPtr]
    call netForward

    mov rdi, [netPtr]
    mov rsi, [currentBuffPtry]
    call netBackward

    movss xmm0, [weightDecay]
    mov rdi, [netPtr]
    call netDecay

    /* keep track of the loss */
    mov rdi, [netPtr]
    mov rax, [rdi]
    lea rax, [rdi+8*rax+8]
    movss xmm0, [rax]
    movss xmm1, [currentLoss]
    addss xmm1, xmm0
    movss [currentLoss], xmm1

    /* step */
    movss xmm0, [currentLR]
    movss xmm1, [momentumGamma]
    call netStep

    call emitTrainStatus

    add r15, 1
    cmp r15, [currentMinibatchesPerEpoch]
    jnz runEpoch_mbl

    mov rdi, SYS_STDOUT
    call printnl

    pop r15
    pop r13
    ret

emitxofyW:
    push r13
    push r14
    push r15
    mov r14, rdi
    mov r15, rsi

    mov rdi, rsi
    call dlogten
    mov r13, rax
   
    mov rsi, r14
    mov rdi, SYS_STDOUT
    mov rdx, r13
    call printIntZPad

    call printsp
    mov sil, '/'
    call printchar
    call printsp

    mov rsi, r15
    mov rdx, r13
    call printIntZPad

    pop r15
    pop r14
    pop r13
    ret

emitTestStatus:
    mov rax, [currentIter]
    add rax, 1
    xor rdx, rdx
    mov rcx, iterPrintRate
    div rcx
    cmp rdx, 0
    jnz 3f
    emitString statStringTe1

    mov rdi, [currentEpochs]
    inc rdi
    mov rsi, [numEpochs]
    call emitxofyW

    emitString statStringTr2

    mov rdi, [currentIter]
    inc rdi
    mov rsi, [currentMinibatchesPerEpoch]
    call emitxofyW
  
    call printnl
    
3:  ret


emitTrainStatus:

    mov rax, [currentIter]
    add rax, 1
    xor rdx, rdx
    mov rcx, iterPrintRate
    div rcx
    cmp rdx, 0
    jnz 3f

    emitString statStringTr1
    mov rdi, [currentEpochs]
    inc rdi
    mov rsi, [numEpochs]
    call emitxofyW

    emitString statStringTr2

    mov rdi, [currentIter]
    inc rdi
    mov rsi, [currentMinibatchesPerEpoch]
    call emitxofyW

    emitString statStringTr3

    movss xmm0, [currentLR]
    mov rdi, SYS_STDOUT
    call printFloat

    emitString statStringTr4

    movss xmm0, [currentLoss]
    mov eax, iterPrintRate
    cvtsi2ss xmm1, eax
    divss xmm0, xmm1
    mov dword ptr [currentLoss], 0

    mov rdi, SYS_STDOUT
    call printFloat

    call printnl
    
3:  ret




identityperm:
    /* rdi -> N
       rsi -> quad *
       i rcx
       Useful debugging tool
    */

    xor rax, rax
1:  mov qword ptr [rsi], rax
    add rax, 1
    add rsi, 8
    sub rdi, 1
    jnz 1b

    ret



randperm:
    /* rdi -> N         -> r12
       rsi -> quad *    -> r13
              i         -> r14
       Fisher-Yates
    */
    push r12
    push r13
    push r14

    mov r12, rdi
    mov r13, rsi

    xor r14, r14
1:
    /* get a number j: 0 <= j <= i/r14 */
    mov rdi, r14
    add rdi, 1
    call randunifi

    cmp rax, r14
    je 2f
    //a[i] <- a[j]
    mov rcx, [r13+8*rax]
    mov [r13+8*r14], rcx
2:  mov [r13+8*rax], r14
    
    add r14, 1
    cmp r14, r12
    jnz 1b

    pop r14
    pop r13
    pop r12
    ret


randunifi:
    /* return an integer from 0, .. rdi-1 inclusive */
    push r12
    mov r12, rdi
    call randqword
    /* random quad in rax */
    xor rdx, rdx
    div r12
    /* save remainder -- we'll probably return this */
    mov rcx, rdx
    /* If overflow redo: it's the last [0,rdi) bucket so it's biased */
    add rax, 1
    mul r12
    jnc 1f
    /* try again */
    mov rdi, r12
    call randunifi
    mov rcx, rax
1:
    mov rax, rcx
    pop r12
    ret


rfill:
    /* rsi -> float * -> r12
       rdi -> N       -> r13
       xmm0 -> mean   -> r14
       xmm1 -> std    -> r15

    Fill: rsi:rsi+N floats with ~N(xmm0,xmm1) noise
     */
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi
    mov r13, rsi
    movd r14, xmm0
    movd r15, xmm1
1:  
    call randnorm
    movd xmm1, r14
    movd xmm2, r15
    mulss xmm0, xmm2
    addss xmm0, xmm1

    movss [r12], xmm0
    add r12, 4
    sub r13, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12
    ret



randnorm:
    /* Marsaglia polar method */
    push r13
    push r14
    push r15
    
1:
    call randunif
    movd r15, xmm0
    call randunif
    movd xmm1, r15

    /* 0,1 \in [0,1] -> 0,1 \in [-1,1] */
    addss xmm0, xmm0
    addss xmm1, xmm1
    subss xmm0, [m_1]
    subss xmm1, [m_1]

    /* got two of them (x,y) in (xmm0, xmm1), compute s = x^2 + y^2 in xmm2 */
    movss xmm2, xmm0
    movss xmm3, xmm1
    mulss xmm2, xmm2
    mulss xmm3, xmm3
    addss xmm2, xmm3
    
    comiss xmm2, [m_1]
    ja 1b

    /* save */
    movd r13, xmm0
    movd r14, xmm1
    movd r15, xmm2
    
    movss xmm0, xmm2
    call ln
    movss xmm3, xmm0
    movd xmm0, r13
    movd xmm1, r14
    movd xmm2, r15
    /* 0=x,1=y,2=s,3=ln(s) */
    mulss xmm3, [m_n2]
    sqrtss xmm3, xmm3
    sqrtss xmm2, xmm2
    divss xmm3, xmm2
    mulss xmm0, xmm3
    mulss xmm1, xmm3

    pop r15
    pop r14
    pop r13
    ret

randunif:
    call randdword
    //1 (float) -> ecx
    mov ecx, 0x3f800000
    and eax, 0x07FFFFF
    add eax, ecx
    //now eax is some random number between [1,2]
    //df: don't be oversmart: you always have a 1.XYZ format, so you probably 
    //can't easily avoid the sub
    movd xmm0, eax
    movd xmm1, ecx
    subss xmm0, xmm1
    ret


randqword:
    mov rdi, 8
    jmp randbytes

randdword:
    mov rdi, 4
    jmp randbytes

randbytes:
    /* return up to rdi bytes in rax for rdi <= 8 
       this uses the honor system that you won't peek at the high bytes 
     */
    push r15
    mov r15, rdi
    mov rdx, [randBufferPtr]
    add rdx, 8
    cmp rdx, RAND_BUFF_SIZE
    jl 1f
    /* otherwise we have to refresh the buffer */
    call randrefillbuff 
    mov rdx, [randBufferPtr]
1:
    mov rax, qword ptr [rdx+byteReserve]
    add qword ptr [randBufferPtr], r15
    pop r15
    ret

randrefillbuff:
    /* refill the buffer 
        r15 file id 
    */

    push r15
    /*open */
    mov rax, SYS_OPEN
    lea rdi, [devRandom]
    xor rsi, rsi
    xor rdx, rdx
    syscall
    mov r15, rax

    /* read in */
    mov rax, SYS_READ
    mov rdi, r15
    lea rsi, [byteReserve]
    mov rdx, RAND_BUFF_SIZE
    syscall

    mov rax, SYS_CLOSE
    mov rdi, r15
    syscall

    /* reset buffer pointer */
    mov qword ptr [randBufferPtr], 0
    pop r15
    ret


/* 
   I HATE STRINGS BEEP BOOP WHY CAN'T PEOPLE SEND PROTOBUFS 
   
   Nice feature -- printchar/printFloat/printInt etc. save rdi
 
 */


strlen:
    /* save, will need to calcualte */
    mov rdx, rdi
    /* 0xFF...FF ->rcx */
    xor rcx, rcx
    dec rcx
    /* scan for 0 */
    xor rax, rax
    repne scasb
    mov rax, rdi
    sub rax, rdx
    ret

/* Silly routines for parsing prototext input */

scanToNum:
    /* given rdi pointer to string return the offset to the next number char */
    xor rax, rax
1:  movzx rcx, byte ptr [rdi+rax]
    cmp cl, 0
    je 2f
    mov cl, byte ptr [chrtype+rcx]
    cmp cl, 1
    je 2f
    add rax, 1
    jmp 1b
2:  ret

nextChunk:
    /* rdi is a null-terminated string starting with a number, return: 
        #chars that are nummish (i.e., callee should clip after this) -> rax
        start of next numerical (i.e., callee should forward to this) -> rdx
    */
    
    xor rax, rax
1:  movzx rcx, byte ptr [rdi+rax]
    mov cl, byte ptr [chrtype+rcx]
    cmp cl, 1
    jnz 2f
    add rax, 1
    jmp 1b
    
2:  mov r9, rax

1:  movzx rcx, byte ptr [rdi+rax]
    mov cl, byte ptr [chrtype+rcx]
    cmp cl, 1
    jz 2f
    add rax, 1
2:  mov rdx, rax
    mov rax, r9

    /* return two values because we can */
    ret


banner:
    push rdi
    /* print the banner to the file id'd by rdi */
    mov eax, SYS_WRITE
    lea rsi, [bannerString]
    mov rdx, bannerStringLen
    syscall

    pop rdi
    ret

printchar:
    /* print char in low byte of rsi to file id'd by rdi */
    push rbp 
    push rdi
    mov rbp, rsp
    sub rsp, 8

    mov eax, SYS_WRITE
    /* rdi is already set */
    mov byte ptr [rbp-8], sil
    lea rsi, [rbp-8]
    mov edx, 1
    syscall

    add rsp, 8
    pop rdi
    pop rbp
    ret

printnl:
    /* print a newline to the file in rdi */
    mov sil, 0xa
    jmp printchar

printsp:
    /* print a newline to the file in rdi */
    mov sil, ' '
    jmp printchar


printFloat:
    /* super dirty print a float from xmm0 to the file in rdi 
       we'll also use knowledge of what printchar and syscalls clobber */
    push rbp
    push rbx
    push r12
    push rdi
    mov rbp, rsp
    sub rsp, 16

    movd edx, xmm0

    /* check for shifty numbers that have an exponent of 0xFF, print them, and 
       then go to the epilogue */
    mov eax, edx
    and eax, 0x7f800000
    cmp eax, 0x7f800000
    jnz printFloat_valid

    /* test if it's a nan or an inf */
    test edx, 0x7fffff
    jz 1f
    mov eax, SYS_WRITE
    lea rsi, [NaNStr]
    mov rdx, 3
    syscall
    jmp 3f
1:  /* but what type of inf */
    bt edx, 31
    jc 1f
    /* rdi preset */
    mov sil,'+'
    call printchar
    jmp 2f
1:  /* rdi preset */
    mov sil, '-'
    call printchar 
2:  mov eax, SYS_WRITE
    lea rsi, [InfStr]
    mov rdx, 3
    syscall
    
3: jmp printFloat_epilogue

printFloat_valid:

    /* stash mxcsr and a copy that we'll muck with */
    stmxcsr [rbp-8]
    stmxcsr [rbp-16]

    /* set round down mode by setting bits 14,13 to 0,1*/
    mov eax, [rbp-16]
    and ax, 0x9fff
    or ax, 0x2000
    mov [rbp-16], eax
    ldmxcsr [rbp-16]

    /* check to see if we print a sign */
    comiss xmm0, [m_0]
    jae printFloat_pos
    negatexmm xmm0, eax
    mov sil, 0x2d
    call printchar

printFloat_pos:
    /* we are now printing a positive float */
    movss xmm1, xmm0
    movss xmm2, [m_0p1]

    /* use rbx so we don't have to preserve */
    xor ebx, ebx
countdigits:
    comiss xmm0, [m_1]
    jnae printFloat_donecounting
    inc rbx
    divss xmm0, [m_10]
    mulss xmm2, [m_10]
    jmp countdigits

printFloat_donecounting:

    /* special case of only fractional, print 0. */
    cmp rbx, 0
    jne printFloat_noleadzero 
    mov sil, 0x30
    call printchar
    mov sil, 0x2e
    call printchar

printFloat_noleadzero:
    /* rbx has number of digits to the left of the decimal and xmm2 = 10^rbx 
       6 sig figs because */
    mov r12, rbx
    add rbx, 6

    movss xmm0, xmm1
printFloat_loop:
    movss xmm3, xmm0
    divss xmm3, xmm2
    cvtss2si rax, xmm3
    cvtsi2ss xmm3, rax
    mov sil, al
    /* oh man such hacks */
    cmp sil, 10
    jl 1f
    mov sil, 9
1:  add sil, 0x30
    call printchar
    mulss xmm3, xmm2
    subss xmm0, xmm3
    divss xmm2, [m_10]

    /* how many until decimal */
    sub r12, 1
    jnz printFloat_reloop
    mov sil, 0x2e
    call printchar

printFloat_reloop:
    sub rbx, 1
    jnz printFloat_loop

    /* restore state */
    ldmxcsr [rbp-8]


printFloat_epilogue:
    add rsp, 16
    pop rdi
    pop r12
    pop rbx
    pop rbp
    ret

printInt:
    mov rdx, 0
    jmp printIntZPad

printIntZPad:
    /*  print the int in rsi into fid rdi with at least
        rcx digits 
   
        rdi -> file id          -> r14
        rsi -> int to print     -> r15
               buffer ptr       -> r13                                
               minimum length to print -> r12
     */
    push rbp
    push rdi
    push rbx
    push r12
    push r14
    push r15

    mov rbp, rsp
    sub rsp, 24
    mov rbx, rbp
    sub rbx, 2
    mov r12, rdx

    mov r14, rdi
    mov r15, rsi 

    /* a 64-bit register can only be 24 decimal digits long */
    cmp r12, 24
    jle 2f
    mov r12, 24
2:

    /* pre-set the buffer to '0' */
    mov al, '0'
    lea rdi, [rbp-24]
    mov rcx, 24
    rep stosb

    /* print signs if needed */
    mov rsi, r15
    cmp rsi, 0
    jge 2f
    /* rdi set */
    mov sil, '-'
    mov rdi, r14
    call printchar
    neg r15
2:

    /* repeatedly divide out, storing the chars */
    mov r8, 10
    mov rax, r15
    mov rdi, rbx
    xor rcx, rcx

1:  /* invariant, rax has rem num, rdi points at string + 1 */
    sub rdi, 1
    xor rdx, rdx
    div r8
    add rdx, 0x30
    mov [rdi], dl 
    add rcx, 1
    cmp rax, 0
    jnz 1b

    cmp rcx, r12
    jge 2f

    /* ok, if we have a minimum # digits to print, adjust
       rcx and the ptr to print leading zeros */
    mov rax, r12
    sub rax, rcx
    mov rcx, r12 
    sub rdi, rax
2:

    /* write it */
    mov eax, SYS_WRITE
    mov rsi, rdi
    mov rdi, r14
    mov rdx, rcx
    syscall 

    add rsp, 24
    pop r15
    pop r14
    pop r12
    pop rbx
    pop rdi
    pop rbp
    ret


writemattofid:
    /*  rdi -> stream id    -> r12
        rsi -> N            -> r13
        rdx -> K            -> r14
        rcx -> float *M     -> r15
               j ... K      -> rbx
    */
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx

1:  
    mov rbx, r14
2:  movss xmm0, [r15]
    mov rdi, r12
    call printFloat
    call printsp

    add r15, 4
    sub rbx, 1
    jnz 2b

    mov rdi, r12
    call printnl

    sub r13, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret


readmat:
    /* rdi -> char *s -> r12
       rsi -> float *M -> r13
       rdx -> N, number of floats to read  -> r14
           fid -> r15
           buffer ptr -> rbx */
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx

    mov rbp, rsp
    sub rsp, 264

    /* move the buffer start to rbx */
    mov rbx, rsp

    /* open the file */
    mov eax, SYS_OPEN
    mov rdi, r12
    xor esi, esi
    xor edx, edx
    syscall
    mov r15, rax

readmat_rloop:
    /* loop invariants: 
        file pointer is before the next value. may have to chew whitespace
        r13 points to next float to write to
        rdx lists how float to write to
        r14 lists how many floats to read
    */

    /* load it into a buffer */
    mov eax, SYS_READ
    mov rdi, r15
    mov rsi, rbx
    mov edx, 255
    syscall
    mov r8, rax
    /* null terminate the end; we can then rely on the fact that it's null-terminated */
    mov byte ptr [rbx+r8+1], 0

    /* rewind where we are in the file, we'll ffwd as needed once we know how many to ffwd */
    mov eax, SYS_LSEEK
    mov rdi, r15
    mov rsi, r8
    neg rsi
    mov edx, 1
    syscall

    /* now we have up to 255 bytes at rbx, we need to find the end of the next float, scan using rdi */
    lea r9, [chrtype]
    mov rdi, rbx
readmat_wsloop:
    movzxb rax, byte ptr [rdi]
    add rax, r9
    mov al, [rax]
    cmp al, 1
    je readmat_scaf 
    add rdi, 1
    jmp readmat_wsloop

readmat_scaf:
    /* ok now we're at the start of the float with rdi, scan to find the end of the float */
    mov rsi, rdi

readmap_scaf_loop:
    movzxb rax, byte ptr [rdi]
    add rax, r9
    mov al, [rax]
    cmp al, 1
    jne readmap_scaf_done
    add rdi, 1
    jmp readmap_scaf_loop
readmap_scaf_done:

    /* ok, so rsi is the start of the float in the buffer, rdi is the first non float char
     We need to: 
        (1) seek forward rdi-rbx in the file
        (2) scanf starting at rsi (so set *rdi to 0)
     */
    mov byte ptr [rdi], 0
 
    /* calculate rdi-rbx, and save it across calls, popping it back into r8 */
    sub rdi, rbx
    push rdi

    /* call sfloat on the char * at rsi  and store to the target*/
    mov rdi, rsi
    call sfloat
    movss [r13], xmm0
    add r13, 4

    pop r8

    /* seek to next float in file */
    mov eax, SYS_LSEEK
    mov rdi, r15
    mov rsi, r8
    mov edx, 1
    syscall
    
    sub r14, 1
    jnz readmat_rloop

    add rsp, 264
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


sint:
    /* rdi -> char *s */
    xor eax, eax
    xor r8d, r8d

    /* check if neg */
    cmp byte ptr [rdi], 0x2d
    jne sint_rloop
    add rdi, 1
    add r8b, 1

sint_rloop:
    movzxb rcx, byte ptr [rdi]
    cmp cl, 0
    je sint_fin

    /* rax * 10 */
    imul rax, 10
    sub rcx, 0x30
    add rax, rcx
    add rdi, 1
    jmp sint_rloop
sint_fin:

    cmp r8, 0
    je sint_ret
    neg rax
sint_ret:
    ret 


sfloat:
    /* rdi -> char *s */
    /* xmm0 = int part, xmm1 = frac part, xmm2 = 0.1, xmm3 = 10, r8 = sign flip */
    pxor xmm0, xmm0
    pxor xmm1, xmm1
    movss xmm2, m_0p1
    movss xmm3, m_10
    xor r8d, r8d

    /* is it negative */
    cmp byte ptr [rdi], 0x2d
    jne sfloat_irloop
    add r8b, 1
    add rdi, 1

sfloat_irloop:
    movzxb rcx, byte ptr [rdi]
    cmp cl, 0
    je sfloat_fin
    cmp cl, 0x2e
    je sfloat_ifin
  
    mulss xmm0, xmm3
    sub rcx, 0x30
    cvtsi2ss xmm4, ecx
    addss xmm0, xmm4 

    add rdi, 1
    jmp sfloat_irloop

sfloat_ifin:
    /* done reading integer part, and we only fall through here for a dot*/
    add rdi, 1

    /* now read fractional part */
sfloat_frloop:
    movzxb rcx, byte ptr [rdi]
    cmp cl, 0
    je sfloat_fin
    
    sub rcx, 0x30
    cvtsi2ss xmm4, ecx
    mulss xmm4, xmm2
    addss xmm1, xmm4

    divss xmm2, xmm3
    add rdi, 1
    jmp sfloat_frloop

sfloat_fin:
    /* return integer + fractional parts */
    addss xmm0, xmm1

    cmp r8, 0
    je sfloat_ret
    negatexmm xmm0 eax
sfloat_ret:
    ret


/* layer manip macros */

/* given rbx is a layer, put the previous layer in rax, trashing rcx */
.macro prevLayerToRAXtRCX
    mov rax, [rbx+LAYER_OFF_BP]
    mov rcx, [rbx+LAYER_OFF_NUM]
    /* note: rbx = rax + rcx*8 + 8, so prev layer is rax + rcx*8 */
    mov rax, [rax+rcx*8]
.endm

/* given rbx is a layer, put the next layer in rax, trashing rcx */
.macro nextLayerToRAXtRCX
    mov rax, [rbx+LAYER_OFF_BP]
    mov rcx, [rbx+LAYER_OFF_NUM]
    /* note: rbx = rax + rcx*8 + 8, so next layer is rax + rcx*8 + 16 */
    mov rax, [rax+rcx*8+16]
.endm

/* given rbx is a layer, get a pointer to the net diagnostic quad trashing rcx */
.macro netDiagnosticToRAXtRCX
    mov rax, [rbx+LAYER_OFF_BP]
    mov rcx, [rax]
    lea rax, [rax+rcx*8+8]
.endm


zeroDiffBuffer:
    /* given layer rdi, clear the diff buffer */
    mov rax, rdi
    mov rdi, [rax+LAYER_OFF_DIFF]
    mov rcx, [rax+LAYER_OFF_N]
    imul rcx, [rax+LAYER_OFF_F]
    /* recall 0 (int) = 0 (float) */
    xor eax, eax
    rep stosd
    ret

zeroPDiffBuffer:
    /* given layer rdi, clear the diff buffer */
    mov rax, rdi
    mov rdi, [rax+LAYER_OFF_PDIFF]
    mov rcx, [rax+LAYER_OFF_P]
    /* recall 0 (int) = 0 (float) */
    xor eax, eax
    rep stosd
    ret


/* an init takes a null-terminated string specifying the type */
defaultInit:
    ret


actInit:
    /* 
       do an init for an activation that keeps the tensor the same size
        rdi -> a partial layer struct -> rbx
        rsi -> string spec -> r15
    */
    push r13
    push r14
    push r15
    push rbx

    mov rbx, rdi
    mov r15, rsi

    /* ok figure out the previous layer */
    prevLayerToRAXtRCX
    /* rax points to previous layer, just steal it */
    mov r13, [rax+LAYER_OFF_N]
    mov [rbx+LAYER_OFF_N], r13

    mov r13, [rax+LAYER_OFF_K]
    mov [rbx+LAYER_OFF_F], r13
    mov [rbx+LAYER_OFF_K], r13

    mov r13, [rbx+LAYER_OFF_N]
    imul r13, [rbx+LAYER_OFF_K]
    shl r13, 2
   
    mov rdi, r13
    call alloc
    mov [rbx+LAYER_OFF_DAT], rax

    mov rdi, r13
    call alloc
    mov [rbx+LAYER_OFF_DIFF], rax

    pop rbx
    pop r15
    pop r14
    pop r13
    ret

linearInitFromStr:
    /* 
        rdi -> layer struct     -> rbx
        rsi -> string spec      -> r15
               offset into str  -> r14
       returns in rax the exponent on the weight size

       separate functions make it easier to keep track of what to trash 
     */
    push rbx
    push r14
    push r15
    mov rbx, rdi
    mov r15, rsi
    xor r14, r14

    mov rdi, r15
    call scanToNum
    mov r14, rax

    lea rdi, [r15+r14]
    call nextChunk
    lea rdi, [r15+r14]
    mov byte ptr [rdi+rax], 0
    add r14, rdx

    /* rdi set before */
    call sint
    mov [rbx+LAYER_OFF_K], rax

    lea rdi, [r15+r14]
    call nextChunk
    lea rdi, [r15+r14]
    mov byte ptr [rdi+rax], 0

    /* rdi set before */
    call sint
    /* return in rax */

    pop r15
    pop r14
    pop rbx
    ret

linearInit:
    /* given
        rdi -> a partial layer struct -> rbx
        rsi -> string spec -> r15
               FxK+K       -> r13
               intialization mag -> r12
        r14 call safe scratch space
    */
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov rbx, rdi
    mov r15, rsi

    call linearInitFromStr
    mov r12, rax

    /* first, figure out N, F */
    prevLayerToRAXtRCX

    mov r13, [rax+LAYER_OFF_N]
    mov [rbx+LAYER_OFF_N], r13

    mov rcx, [rax+LAYER_OFF_K]
    mov [rbx+LAYER_OFF_F], rcx
    /* ok done using previous layer */

    /* allocate (F*K+K) parameters */
    mov r13, [rbx+LAYER_OFF_F]
    add r13, 1
    imul r13, [rbx+LAYER_OFF_K]

    mov [rbx+LAYER_OFF_P], r13

    /* ok alloc param stuff */
    mov r14, r13
    shl r14, 2

    mov rdi, r14
    call alloc
    mov [rbx+LAYER_OFF_PDAT], rax

    mov rdi, r14
    call alloc
    mov [rbx+LAYER_OFF_PDIFF], rax

    mov rdi, r14
    call alloc
    mov [rbx+LAYER_OFF_PMOM], rax

    /* now allocate output data */
    mov rdi, [rbx+LAYER_OFF_N]
    imul rdi, [rbx+LAYER_OFF_K]
    shl rdi, 2
    call alloc
    mov [rbx+LAYER_OFF_DAT], rax

    /* allocate bottom diff */
    mov rdi, [rbx+LAYER_OFF_N]
    imul rdi, [rbx+LAYER_OFF_F]
    shl rdi, 2
    call alloc
    mov [rbx+LAYER_OFF_DIFF], rax

    /* initialize the weights */

    mov rdi, r12
    call tenee
    movss xmm1, xmm0
    pxor xmm0, xmm0
    mov rdi, [rbx+LAYER_OFF_PDAT]
    mov rsi, r13
    /* (rdi...rdi+r13 floats) ~ N(xmm0,xmm1) */
    call rfill

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    ret


inputInitFromStr:
    /* given rdi layer      -> rbx
             rsi string     -> r15
             offset in str  -> r14
        returns nothing
    */
    push rbx
    push r14
    push r15
    mov rbx, rdi
    mov r15, rsi
    xor r14, r14

    mov rdi, r15
    call scanToNum
    mov r14, rax

    lea rdi, [r15+r14]
    call nextChunk
    lea rdi, [r15+r14]
    mov byte ptr [rdi+rax], 0
    add r14, rdx

    /* rdi set before */
    call sint
    mov [rbx+LAYER_OFF_K], rax

    lea rdi, [r15+r14]
    call nextChunk
    lea rdi, [r15+r14]
    mov byte ptr [rdi+rax], 0

    /* rdi set before */
    call sint
    mov [rbx+LAYER_OFF_N], rax

    pop r15
    pop r14
    pop rbx
    ret


inputInit:
    /* given:
       rdi -> a partial layer struct -> rbx
       rsi -> string spec -> r15
              N            -> r13
              K            -> r14
     
     */
    push r13
    push r14
    push r15
    push rbx

    mov rbx, rdi
    mov r15, rsi

    call inputInitFromStr

    mov r13, [rbx+LAYER_OFF_N]
    mov r14, [rbx+LAYER_OFF_K]

    /* calculate data size */
    mov rdi, r13
    imul rdi, r14
    shl rdi, 2
    call alloc
    mov [rbx+LAYER_OFF_DAT], rax

    mov rdi, [rbx+LAYER_OFF_DAT]
    mov rsi, [rbx+LAYER_OFF_N]
    imul rsi, [rbx+LAYER_OFF_K]
    call ofill

    pop rbx
    pop r15
    pop r14
    pop r13
    ret

/* a forward maps the input blob to the output blob */
defaultForward:
    ret


copyForward:
    /* maps the activations forward */
    push rbx
    mov rbx, rdi
    prevLayerToRAXtRCX

    mov rsi, [rax+LAYER_OFF_DAT]
    mov rdi, [rbx+LAYER_OFF_DAT]
    mov rcx, [rbx+LAYER_OFF_N]
    imul rcx, [rbx+LAYER_OFF_K]
    
    xor rdx, rdx
1:  mov eax, [rsi+rdx]
    mov [rdi+rdx], eax
    add rdx, 4
    sub rcx, 1
    jnz 1b

    pop rbx
    ret


softmaxForward:
    push rbx
    mov rbx, rdi
    call copyForward

    /* setup softmax call */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_K]
    mov rdx, [rbx+LAYER_OFF_DAT]
    call softmaxRowwise

    pop rbx
    ret

sceForward:
    push rbx
    mov rbx, rdi
    call copyForward

    /* setup softmax call */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_K]
    mov rdx, [rbx+LAYER_OFF_DAT]
    call softmaxRowwise

    pop rbx
    ret

linearForward:
    /* given a linear layer pointed to in rdi, do the forward pass */
    push rbx
    mov rbx, rdi
    prevLayerToRAXtRCX

    /* setup the mmul call */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_F]
    mov rdx, [rbx+LAYER_OFF_K]

    /* *rcx = multiply(*r8,*r9) */
    mov rcx, [rbx+LAYER_OFF_DAT]
    mov r8, [rax+LAYER_OFF_DAT]
    mov r9, [rbx+LAYER_OFF_PDAT]

    call mmul

    /* now add the biases */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_K]
    mov rdx, [rbx+LAYER_OFF_DAT]

    /* calculate bias location */
    mov rax, [rbx+LAYER_OFF_F]
    imul rax, [rbx+LAYER_OFF_K]
    mov rcx, [rbx+LAYER_OFF_PDAT]
    lea rcx, [rcx+4*rax]
    call broadcastmv

    pop rbx
    ret

reluForward:
    /* given a relu layer pointed to in rdi, do the forward pass */
    push rbx
    mov rbx, rdi
    prevLayerToRAXtRCX
    mov rsi, [rax+LAYER_OFF_DAT]
    mov rdi, [rbx+LAYER_OFF_DAT]
    mov rcx, [rbx+LAYER_OFF_N]
    imul rcx, [rbx+LAYER_OFF_K]

    pxor xmm1, xmm1
1:  movss xmm0, [rsi]
    maxss xmm0, xmm1 
    movss [rdi], xmm0
    add rsi, 4
    add rdi, 4
    sub rcx, 1
    jnz 1b

    pop rbx
    ret

tanhForward:
    /* given a tanh layer pointed to in rdi, do the forward pass
        rdi ->  layer ptr       -> rbx
                prev layer act  -> r12
                out act         -> r13
                counter         -> r14

     */
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbx, rdi

    prevLayerToRAXtRCX
    mov r12, [rax+LAYER_OFF_DAT]
    mov r13, [rbx+LAYER_OFF_DAT]
    mov r14, [rbx+LAYER_OFF_N]
    imul r14, [rbx+LAYER_OFF_K]
    
1:  movss xmm0, [r12]
    call tanh
    movss [r13], xmm0
    add r12, 4
    add r13, 4
    sub r14, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12 
    pop rbx
    ret



/* a backward computes the bottom diff */
defaultBackward:
    ret


reluBackward:
    /* rdi -> layer point       -> rbx

    */
    push rbx
    mov rbx, rdi

    nextLayerToRAXtRCX
    mov rsi, rax
    mov rsi, [rsi+LAYER_OFF_DIFF]

    prevLayerToRAXtRCX
    mov rdx, rax
    mov rdx, [rdx+LAYER_OFF_DAT]

    mov rdi, [rbx+LAYER_OFF_DIFF]

    /* *rdi += *rsi if *rdx > 0 else 0 */
    pxor xmm3, xmm3
    xor rax, rax

    mov rcx, [rbx+LAYER_OFF_N]
    imul rcx, [rbx+LAYER_OFF_K]

1:  movss xmm2, [rdx+rax]
    comiss xmm2, xmm3
    jna 2f
    /* positive */ 
    movss xmm0, [rdi+rax]
    addss xmm0, [rsi+rax]
    movss [rdi+rax], xmm0 
2:
    add rax, 4
    sub rcx, 1
    jnz 1b
    
    pop rbx
    ret

tanhBackward:
    /* rdi -> layer ptr         -> rbx
              prev layer act    -> r12  
              next layer deriv  -> r13
              this layer deriv  -> r14
              NxK               -> r15
    */
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbx, rdi

    nextLayerToRAXtRCX
    mov r13, rax
    mov r13, [r13+LAYER_OFF_DIFF]
    
    prevLayerToRAXtRCX
    mov r12, rax
    mov r12, [rbx+LAYER_OFF_DAT]

    mov r14, [rbx+LAYER_OFF_DIFF]

    mov r15, [rbx+LAYER_OFF_N]
    imul r15, [rbx+LAYER_OFF_K]

    /* *r14 = 1 -(*r12)^2 * *r13 
        dtanh(x) = 1-tanh(x)^2
     */
    movss xmm2, [m_1]

1:  movss xmm1, [r12]
    mulss xmm1, xmm1
    movss xmm0, xmm2
    subss xmm0, xmm1
    mulss xmm0, [r13]
    movss [r14], xmm0

    add r12, 4
    add r13, 4
    add r14, 4
    sub r15, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
    
        

linearBackward:
    /* rdi -> layer pointer     -> rbx
              prev layer act    -> r12 
              next layer deriv  -> r13 

    */

    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp

    mov rbx, rdi

    nextLayerToRAXtRCX
    mov r13, rax
    mov r13, [r13+LAYER_OFF_DIFF]

    prevLayerToRAXtRCX
    mov r12, rax
    mov r12, [r12+LAYER_OFF_DAT]

    /* first compute the backwards for W */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_F]
    mov rdx, [rbx+LAYER_OFF_K]
    mov rcx, [rbx+LAYER_OFF_PDIFF]
    mov r8, r12
    mov r9, r13
    call smtm

    /* now compute backwards of biases */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_K]

    /* calculate bias location */
    mov rdx, [rbx+LAYER_OFF_PDIFF]
    mov rax, [rbx+LAYER_OFF_F]
    imul rax, [rbx+LAYER_OFF_K]
    lea rdx, [rdx+4*rax]
    mov rcx, r13
    call broadcastdownvm


    /* compute backwards to prev layer */
    mov rdi, [rbx+LAYER_OFF_N]
    mov rsi, [rbx+LAYER_OFF_F]
    mov rdx, [rbx+LAYER_OFF_K]
    mov rcx, [rbx+LAYER_OFF_DIFF]
    mov r8, r13
    mov r9, [rbx+LAYER_OFF_PDAT]
    call smmt

    pop rbp
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
    

l2backward:
    /* given rdi layer and rsi pointer to float gts, compute backwards */
    push rbx
    push r12

    mov rbx, rdi
    mov r12, rsi

    prevLayerToRAXtRCX

    mov r9, [rbx+LAYER_OFF_N]
    mov rdi, [rbx+LAYER_OFF_DIFF]
    mov rsi, [rax+LAYER_OFF_DAT]

    /* average loss over samples */
    pxor xmm3, xmm3

    xor rcx, rcx
1:  movss xmm0, [rsi+4*rcx]
    movss xmm1, [r12+4*rcx]
    subss xmm0, xmm1
    movss [rdi+4*rcx], xmm0 

    /* add (y-yh)^2 */
    mulss xmm0, xmm0
    addss xmm3, xmm0 

    add rcx, 1
    cmp rcx, r9
    jnz 1b

    /* display the current l2 loss */
    cvtsi2ss xmm2, r9
    movss xmm0, xmm3
    divss xmm0, xmm2

    /* derefence to get net to set diagnostic */
    netDiagnosticToRAXtRCX
    movss [rax], xmm0

    pop r12
    pop rbx
    ret

sceBackward:
    /* 
     given rdi layer and rsi pointer to float gts, compute backwards
     r14, r15 used to preserve xmm3, xmm1 
     */
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbx, rdi
    mov r12, rsi


    mov r9, [rbx+LAYER_OFF_N]
    mov r10, [rbx+LAYER_OFF_K]
    mov r11, r9
    imul r11, r10
    mov rdi, [rbx+LAYER_OFF_DIFF]
    mov rsi, [rbx+LAYER_OFF_DAT]

    /* copy back */
    xor rcx, rcx
1:  movss xmm0, [rsi+4*rcx]
    movss [rdi+4*rcx], xmm0
    add rcx, 1
    cmp rcx, r11
    jnz 1b

    /* handle the GT */
    movss xmm1, [m_n1]
    pxor xmm3, xmm3

    xor rcx, rcx
1:  /* identify the class num */
    movss xmm0, [r12+4*rcx]
    xor rdx, rdx
    cvtss2si edx, xmm0
    movss xmm0, [rsi+4*rdx]
    /*stash it away in xmm2, we'll use this to update loss */
    movss xmm2, xmm0
    addss xmm0, xmm1
    movss [rdi+4*rdx], xmm0

    /* Alternate -- report p(gt) 
    addss xmm3, xmm2
    and comment the below mess
       */

    /* All this work for a freaking log. TODO: re-think register usage above 
       to reduce pushes a bit. Could depend on known clobbers but that's a bug 
       I don't want to track down. Also when you debug this remember: 
       log(\sum) can be very different than \sum(log) if the values are varying
    */
    push rcx
    push r9
    push r10
    push rsi
    push rdi
    movd r15, xmm3
    movd r14, xmm1
    movss xmm0, xmm2
    call ln
    negatexmm xmm0 eax
    movd xmm1, r14
    movd xmm3, r15
    pop rdi
    pop rsi
    pop r10
    pop r9
    pop rcx
    addss xmm3, xmm0

    /* increment by a row of K */
    lea rsi, [rsi+4*r10]
    lea rdi, [rdi+4*r10]
    add rcx, 1
    cmp rcx, r9
    jnz 1b

doneloop:
    cvtsi2ss xmm2, r9
    movss xmm0, xmm3
    divss xmm0, xmm2

    /* derefence to get net to set diagnostic */
    netDiagnosticToRAXtRCX
    movss [rax], xmm0

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret


netForward:
    /* given a net rdi, compute the forward pass */
    push rbx
    push r12
    push r15

    mov rbx, rdi
    mov r15, rdi

    mov r12, [rdi]
    sub r12, 1
    add rbx, 16
1:  mov rdi, [rbx]
    mov rax, [rdi+LAYER_OFF_ID]
    mov rax, [layerForwardTable+8*rax]
    call rax

    add rbx, 8
    sub r12, 1
    jnz 1b

    pop r15
    pop r12
    pop rbx
    ret


netBackward:
    /* given a net rdi and targets rsi, do backwards */
    push rbx
    push r13
    push r14
    push r15

    mov rbx, rdi
    mov r13, rsi
    mov r15, [rbx]

1:  mov r14, [rbx+8*r15]
    mov rdi, r14
    call zeroDiffBuffer

    mov rdi, r14
    call zeroPDiffBuffer

    mov rdi, r14
    mov rsi, r13
    mov rax, [rdi+LAYER_OFF_ID]
    mov rax, [layerBackwardTable+8*rax]
    call rax

    sub r15, 1
    cmp r15, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop rbx
    ret


initWrapper:
    /* given: 
       rdi -> type id -> r12
       rsi -> specification string -> r13
       rdx -> pointer to network struct -> r14
       rcx -> layer # -> r15
              pointer to return -> rbx                
       */
    push r12
    push r13
    push r14
    push r15
    push rbx
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    
    mov rdi, LAYER_SIZE
    call alloc
    mov rbx, rax


    /* set some defaults */
    /* clear it */
    mov rcx, LAYER_SIZE
    mov al, 0
    mov rdi, rbx
    rep stosb

    /* do what the stub should do */
    mov [rbx+LAYER_OFF_ID], r12
    mov [rbx+LAYER_OFF_BP], r14
    mov [rbx+LAYER_OFF_NUM], r15
    /* set the pointer in the net struct */
    mov [r14+8*r15+8], rbx

    /* figure out which initializer to call */
    mov rax, [layerInitTable+8*r12]
    mov rdi, rbx
    mov rsi, r13
    call rax

    mov rax, rbx

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    ret

inputIngest:
    /* rdi -> network ptr  -> deref'd immediately to first layer -> rbx
       rsi -> float *  -> r12
       rdx -> reindexer -> r13
       rcx -> num to ingest -> r14
              num cols  -> r15

    Assumes: input is the first layer
             input's input out is same as cols of input matrix
    */
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov rbx, rdi
    mov rbx, [rbx+8]
    mov r12, rsi
    mov r13, rdx
    mov r14, rcx
    mov r15, [rbx+LAYER_OFF_K]
    mov rdi, [rbx+LAYER_OFF_DAT]

    xor r9, r9
1:
    /* figure the actual row in float * from the reindexer */
    mov rax, [r13+8*r9]
    /* how much to add? */
    imul rax, r15
    lea rsi, [r12+4*rax]

    /* do the same, but using r9 r13[r9] */
    mov rax, r9
    imul rax, r15
    mov rdi, [rbx+LAYER_OFF_DAT]
    lea rdi, [rdi+4*rax]
    
    xor rcx, rcx
2:  mov eax, [rsi+4*rcx]
    mov [rdi+4*rcx], eax

    add rcx, 1
    cmp rcx, r15
    jnz 2b

    add r9, 1
    cmp r9, r14
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

getNetworkMinibatchSize:
    /* rdi -> network ptr 
       returns input minibatch count */
    mov rdi, [rdi+8]
    mov rax, [rdi+LAYER_OFF_N]
    ret

getNetworkInputSize:
    /* rdi -> network ptr 
       returns input col count */
    mov rdi, [rdi+8]
    mov rax, [rdi+LAYER_OFF_K]
    ret

getNetworkOutputSize:
    /* rdi -> network ptr 
       returns input col count */
    mov rax, [rdi]
    mov rdi, [rdi+8*rax]
    mov rax, [rdi+LAYER_OFF_K]
    ret


allocNet:
    /* allocate a network with rdi layers and return a pointer to it 
        r14 -> pointer to network struct
        r15 -> holder for num layers
     */
    push r15
    push r14
    mov r15, rdi
    lea rdi, [rdi*8+8]
    call alloc
    mov r14, rax
    mov qword ptr [r14], r15

    mov rcx, r15
    xor rax, rax
    lea rdi, [r14+8]
    rep stosq

    mov rax, r14
    pop r14
    pop r15
    ret

/* local macro; read 255 bytes into the buffer at rbx from file in r15 */
.macro buf_readRBX
    mov eax, SYS_READ
    mov rdi, r15
    mov rsi, rbx
    mov edx, 255
    syscall
.endm

/* local macro: seek r14 bytes from current position in file r15 */
.macro buf_seekR14
    mov eax, SYS_LSEEK
    mov rdi, r15
    mov rsi, r14
    mov edx, 1
    syscall
.endm

parseNet:
    /*  rdi -> char *s filename
        pointer to net  -> r12 
        how many left   -> r13
        helper register -> r14       
        fid             -> r15 
        buffer ptr      -> rbx 
     
      Returns a pointer to the network structure 
     */

    push rbp
    push r12
    push r13
    push r14
    push r15 
    push rbx

    sub rsp, 264
    mov rbx, rsp

   
    /* open the file */
    mov eax, SYS_OPEN
    /* rdi already set */
    xor esi, esi
    xor edx, edx
    syscall
    mov r15, rax

    /* read into buffer */
    buf_readRBX
    mov r14, rax
    neg r14

    /* rewind; r14 is now free for misc */
    buf_seekR14

    /* ok where's the newline */
    mov rdi, rbx
    mov al, '\n'
    mov ecx, 255
    repne scasb

    mov rax, rdi
    sub rax, 1
    mov byte ptr [rax], 0

    /* now rdi points to the newline */
    mov r14, rdi
    sub r14, rbx

    /* forward as much */
    buf_seekR14

    mov rdi, rbx
    call sint
    mov r13, rax

    /* allocate the network */
    mov rdi, r13
    call allocNet
    mov r12, rax
    
parseNet_loop:
    /* invariant: we're always at the start of a line here.  */

    /* read into the buffer, and \n terminate to make seeking easy */ 
    buf_readRBX
    mov byte ptr [rbx+rax+1], '\n'
    mov r14, rax
    neg r14
    
    /* rewind, we'll ffwd as needed */
    buf_seekR14

    /* scan for nl */
    mov rdi, rbx
    mov al, '\n'
    mov ecx, 255
    repne scasb

    mov rax, rdi
    mov byte ptr [rax], 0

    /* forward */
    mov r14, rdi
    sub r14, rbx
    buf_seekR14

    /* we now have set up the loop invariant and we have a buffer rbx with the layer we want to set up */
    mov eax, dword ptr [rbx]
    and eax, 0xffffff
    lea rdi, [layerName]
    mov ecx, LAYER_COUNT
    repne scasq
    neg ecx
    add ecx, LAYER_COUNT
    dec ecx
    /* rcx has the index */

    /* ok initialize layer */
    mov rdi, rcx
    lea rsi, [rbx+4]
    mov rdx, r12
    mov rcx, r13
    neg rcx
    add rcx, [r12]
    call initWrapper

    sub r13, 1
    jnz parseNet_loop

    mov rax, r12
    add rsp, 264
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


/* One way dynamic memory allocation. 
   Turns out, it's much easier if you never free. 
*/ 


alloc:
    /* return a pointer to rdi bytes -- with a tag prefixing it */
     
    push r15
    push r14
    push r13
    mov r13, rdi
    mov r15, rdi 
    add r15, 16

    /* init if we need */
    cmp qword ptr [heapStart], 0
    jnz 1f
    call initHeap
1:  /* resize if we need */
    mov rax, [heapPtr]
    add rax, r15
    cmp rax, [heapSize]
    jle 1f

    mov rdx, [heapSize]
    /* double it until it's big enough */
2:  shl rdx, 1
    cmp rdx, rax
    jl 2b
    mov [heapSize], rdx
    call heapResize

1:  mov rax, [heapPtr]
    mov rdx, rax


    /* return the current heap pointer + heap start */
    add rax, [heapStart]

    /* add the tags */
    mov rsi, 0xcaffe64caffe8664
    mov qword ptr [rax], rsi
    mov [rax+8], r13

    add rax, 16

    /* update the heap pointer */
    add rdx, r15
    mov [heapPtr], rdx

    pop r13
    pop r14
    pop r15
    ret



initHeap:
    /* initialize the heap */
    mov rax, SYS_BRK
    xor rdi, rdi
    syscall
    mov [heapStart], rax
    mov qword ptr [heapSize], INIT_HEAP_SIZE
    call heapResize
    ret

heapResize:
    /* resize the heap to heapSize */ 
    mov rax, SYS_BRK
    mov rdi, [heapStart]
    add rdi, [heapSize]
    syscall 
    ret



/* net format 
network with L layers and P = \sum_i P_i params takes up
8 + 8*L + 4*P = 8 + 8L + 4P bytes
(0) L
(8) P_1
P_1 floats
(P_1*4+8+8) P_2
P_2 floats
((P_1+P_2)*4+8+2*8) P_3
P_3 floats 
..
*/


saveNetFlushParam:
    /* given layer rdi and file id rsi write the params to rsi 
     
        rdi -> layer        -> rbx
                param ptr   -> r13 
                loop        -> r14
        rsi -> fid          -> r15
     */
    push rbp
    push rbx
    push r13
    push r14
    push r15
    mov rbp, rsp
    sub rsp, 8
    mov rbx, rdi
    mov r15, rsi

    mov r14, [rbx+LAYER_OFF_P]
    cmp r14, 0
    jz 3f

    mov r13, [rbx+LAYER_OFF_PDAT]
1:  mov eax, [r13]
    
    mov [rbp-8], eax
    mov rax, SYS_WRITE
    mov rdi, r15
    lea rsi, [rbp-8]
    mov rdx, 4
    syscall 

    add r13, 4
    sub r14, 1
    jnz 1b 


3:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop rbx
    pop rbp
    ret

/* local macro, write a 64-bit register out */
.macro saveNetQuadFlush r64=rax
    mov [rbp-8], \r64
    mov rax, SYS_WRITE
    mov rdi, r15
    lea rsi, [rbp-8]
    mov rdx, 8
    syscall
.endm

saveNet:
    /* rdi -> net ptr           -> rbx
       rsi -> char * filename    
              file id           -> r15
       call-safe scratch        -> r14
    */
    push rbp
    push rbx
    push r14
    push r15
    mov rbp, rsp
    sub rsp, 8

    mov rbx, rdi

    /* open */
    mov eax, SYS_OPEN 
    mov rdi, rsi
    mov esi, 0x0041
    mov edx, 0777
    syscall
    mov r15, rax

    mov rax, [rbx]
    saveNetQuadFlush rax
    
    xor r14, r14
1:  mov rax, [rbx+8*r14+8]
    mov rax, [rax+LAYER_OFF_P]
    saveNetQuadFlush rax

    mov rdi, [rbx+8*r14+8]
    mov rsi, r15
    call saveNetFlushParam

    add r14, 1
    cmp r14, [rbx]
    jnz 1b

    /* close */
    mov eax, SYS_CLOSE
    mov rdi, r15
    syscall

    add rsp, 8
    pop r15
    pop r14
    pop rbx
    pop rbp
    ret

loadNetLayer:
    /* rdi -> layer ptr             -> rbx
       rsi -> file id               -> r15
              P                     -> r14
              counter               -> r13
              flag for store/not    -> r12

    1s loop
    2s skip over for store/no store
    3 jumps to end
    */
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rbp, rsp
    sub rsp, 8
    
    mov rbx, rdi
    mov r15, rsi

    /* read the number of parameters */
    mov eax, SYS_READ
    mov rdi, r15
    lea rsi, [rbp-8]
    mov edx, 8
    syscall
    mov r14, [rbp-8]
    cmp r14, 0
    jz 3f
   
    /* assume don't store */
    mov r12, 1
    cmp r14, [rbx+LAYER_OFF_P]
    jz 2f
    xor r12, r12

    mov eax, SYS_WRITE
    xor rdi, rdi
    lea rsi, [loadSizemismatchStr]
    mov rdx, loadSizemismatchStrLen
    syscall 

2:  

    xor r13, r13
1:  
    /* read a float */
    mov rax, SYS_READ
    mov rdi, r15
    lea rsi, [rbp-8]
    mov edx, 4
    syscall

    mov eax, [rbp-8]
    cmp r12, 0
    jz 2f
    /* store @ P[r13] */
    mov rdi, [rbx+LAYER_OFF_PDAT]
    lea rdi, [rdi+4*r13]
    mov dword ptr [rdi], eax
2:
    add r13, 1
    cmp r13, r14
    jnz 1b

3:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

loadNet:
    /*  rdi -> net ptr              -> rbx
        rsi -> char * filename
                file id             -> r15
                call-safe scratch   -> r14
                num layers in file  -> r12
    */
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov rbp, rsp
    sub rsp, 8
    
    mov rbx, rdi

    mov eax, SYS_OPEN
    mov rdi, rsi
    xor rsi, rsi
    xor edx, edx
    syscall
    mov r15, rax

    /* read the number of layers */
    mov eax, SYS_READ
    mov rdi, r15
    lea rsi, [rbp-8]
    mov edx, 8
    syscall
    mov r12, [rbp-8]

    xor r14, r14
1:  mov rdi, [rbx+8+r14*8]
    mov rsi, r15
    call loadNetLayer

    add r14, 1
    cmp r14, r12
    jnz 1b


    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


addWeightDecay:
    /* given an rdi layer and xmm0 factor, add weight decay */
    push rbx
    mov rbx, rdi

    mov rcx, [rbx+LAYER_OFF_P]

    /* skip if there aren't any to update */
    cmp rcx, 0
    jz 2f

    mov rdi, [rbx+LAYER_OFF_PDIFF]
    mov rsi, [rbx+LAYER_OFF_PDAT]
    mov rax, [rbx+LAYER_OFF_P]

    movss xmm3, xmm0

    xor rax, rax
1:  movss xmm0, [rsi+rax]
    mulss xmm0, xmm3
    addss xmm0, [rdi+rax]
    movss [rdi+rax], xmm0

    add rax, 4
    sub rcx, 1
    jnz 1b

2:
    pop rbx
    ret

netDecay:
    /* given a network in rdi, add weight decay according to factor xmm0 */
    push rbx
    push r12
    push r13
    push r15

    mov rbx, rdi
    mov r15, rdi
    movd r13, xmm0

    mov r12, [rdi]
    /* don't do the input and loss, start at input+1 */ 
    sub r12, 2
    add rbx, 16
1:  
    mov rdi, [rbx]
    movd xmm0, r13
    call addWeightDecay

    add rbx, 8
    sub r12, 1
    jnz 1b

    pop r15
    pop r13
    pop r12
    pop rbx
    ret



sgdStep:
    /* step given:
        rdi layer 
        xmm0 lr
        xmm1 momentum gamma 
     */
    push rbx
    mov rbx, rdi

    mov rcx, [rbx+LAYER_OFF_P]

    /* skip if there aren't any to update */
    cmp rcx, 0
    jz 2f

    mov rdx, [rbx+LAYER_OFF_PMOM]
    mov rsi, [rbx+LAYER_OFF_PDIFF]
    mov rdi, [rbx+LAYER_OFF_PDAT]

    /* lr = xmm3, gamma = xmm4 */
    movss xmm3, xmm0
    movss xmm4, xmm1

    /* divide the gradient by the number of samples */
    mov rax, [rbx+LAYER_OFF_N]
    cvtsi2ss xmm5, eax
    divss xmm3, xmm5  

    xor rax, rax
1:  movss xmm0, [rdi+rax]
    movss xmm1, [rsi+rax]
    movss xmm2, [rdx+rax]

    /* calculate momentum 
       mom = mom * gamma + grad * lr
       xmm2/rdx = xmm2/rdx * xmm4 + xmm1/rsi * xmm3
    */
    mulss xmm2, xmm4
    mulss xmm1, xmm3
    addss xmm2, xmm1
    /* store momentum, take a step, store */
    movss [rdx+rax], xmm2
    subss xmm0, xmm2
    movss [rdi+rax], xmm0

    add rax, 4
    sub rcx, 1
    jnz 1b

2:
    pop rbx
    ret


netStep:
    /* given a network in rdi, take a step according to learning rate xmm0 
       and momentum xmm1
     */
    push rbx
    push r12
    push r13
    push r14
    push r15

    movd r13, xmm0
    movd r14, xmm1

    mov rbx, rdi
    mov r15, rdi

    mov r12, [rdi]
    /* don't do the input and loss, start at input+1 */ 
    sub r12, 2
    add rbx, 16
1:  
    mov rdi, [rbx]
    movd xmm0, r13
    movd xmm1, r14
    call sgdStep

    add rbx, 8
    sub r12, 1
    jnz 1b

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

layerOptimInit:
    /* initialize the optimization parameters of a layer rdi */
    push rbx
    mov rbx, rdi

    xor eax, eax
    mov rdx, [rbx+LAYER_OFF_P]

    cmp rdx, 0
    jz 1f
  
    mov rcx, rdx
    mov rdi, [rbx+LAYER_OFF_PDIFF]
    rep stosd

    mov rcx, rdx
    mov rdi, [rbx+LAYER_OFF_PMOM]
    rep stosd

1:
    pop rbx
    ret

netOptimInit:
    /* initialize the optimization parameters of a network rdi */
    push rbx
    push r12

    mov rbx, rdi

    mov r12, [rdi]
    sub r12, 2
    add rbx, 16
1:  mov rdi, [rbx]
    call layerOptimInit

    add rbx, 8
    sub r12, 1
    jnz 1b

    pop r12
    pop rbx
    ret


/* Does it look like I want to write BLAS? C'mon */



broadcastmv:
    /* rdi -> int N -> r8 (decreasing)
       rsi -> int K -> r10 (decreasing, loop var r9)
       rdx -> float *M (NxK) -> rdi
       rcx -> float *v (1xK) -> r11, reset to rsi
       for i ... N
           M(i,:) += v 
    */

    mov r8, rdi
    mov r10, rsi
    mov rdi, rdx
    mov r11, rcx
   
broadcastmv_NLoop:

    mov r9, r10
    mov rsi, r11
broadcastmv_KLoop:
    movss xmm0, [rdi]
    addss xmm0, [rsi]
    movss [rdi], xmm0
    add rsi, 4
    add rdi, 4
    sub r9, 1
    jnz broadcastmv_KLoop

    sub r8, 1
    jnz broadcastmv_NLoop

    ret

broadcastdownvm:
    /*  rdi -> int N -> r8 (decreasing)
        rsi -> int K -> r10 (decreasing, loop var v9)
        rdx -> float *v (1xK) -> r11, reset to rdi
        rcx -> float *M (NxK) -> rsi, 
        for i, ... N
            v += M(i,:)
    */
    mov r8, rdi
    mov r10, rsi
    mov r11, rdx
    mov rsi, rcx

broadcastdownvm_NLoop:

    mov r9, r10
    mov rdi, r11
broadcastdownvm_KLoop:
    movss xmm0, [rdi]
    addss xmm0, [rsi]
    movss [rdi], xmm0

    add rsi, 4
    add rdi, 4
    sub r9, 1
    jnz broadcastdownvm_KLoop

    sub r8, 1
    jnz broadcastdownvm_NLoop

    ret


smmt:
    /* Sum M M transpose 
       rdi -> int N -> r12
       rsi -> int F -> r13
       rdx -> int K -> r14
       rcx -> float *d (NxF) -> r15
       r8  -> float *dy (NxK) -> rbx
       r9 -> float *W (FxK) -> rbp

    Set:
      d += dy * W'
    */
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8
    mov rbp, r9

    /* loop counters r8, r9, r10 */
    xor r8, r8
smmt_NLoop:


    xor r9, r9 
smmt_FLoop:
    /* this is surprisingly more civilized since we're multiplying the rows 
       of dy with the columns of W^T aka rows of W */
   
    mov rax, r14
    imul rax, r8
    lea rsi, [rbx+4*rax]

    mov rax, r14
    imul rax, r9
    lea rdi, [rbp+4*rax]

    pxor xmm0, xmm0
    xor r10, r10
smmt_KLoop:
    movss xmm1, [rsi]
    mulss xmm1, [rdi]
    addss xmm0, xmm1
    add rsi, 4
    add rdi, 4
    add r10, 1
    cmp r10, r14
    jnz smmt_KLoop


    /* d (NxF) so * = [r8*F+r9] */
    mov rax, r8
    imul rax, r13
    add rax, r9

    lea rax, [r15+4*rax]
    movss xmm1, [rax]
    addss xmm0, xmm1
    movss [rax], xmm0


    add r9, 1
    cmp r9, r13
    jnz smmt_FLoop

    add r8, 1
    cmp r8, r12
    jnz smmt_NLoop

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret



smtm:
    /* 
       Sum M transpose M
       
       rdi -> int N -> r12
       rsi -> int F -> r13
       rdx -> int K -> r14
       rcx -> float *dw (FxK) -> r15
       r8  -> float *X (NxF) -> rbx
       r9  -> float *dy (NxK) -> rbp

       Set:
        dw += X'*dy

    */
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8
    mov rbp, r9

    /* loop counters r8, r9, r10 */

    xor r8, r8
smtm_FLoop: 


    xor r9, r9
smtm_KLoop:
    /* ok have to compute the dot product of columns r8, r9
       rsi + increments of rcx will handle r8
       rdi + increments of rdx will handle r9
       r10 counts
     */
    lea rsi, [rbx+4*r8]
    mov rcx, r13
    shl rcx, 2

    lea rdi, [rbp+4*r9]
    mov rdx, r14
    shl rdx, 2

    pxor xmm0, xmm0
    mov r10, r12
smtm_dploop:
    movss xmm1, [rsi]
    mulss xmm1, [rdi]
    addss xmm0, xmm1
    add rsi, rcx
    add rdi, rdx
    sub r10, 1
    jnz smtm_dploop

    /* ok xmm0 = X[:,r8] * dy[:,r9], sum, figure out where to stick it
       in dw and accumulate */

    /* r8 is ...F, r9 is ...K; dw is FxK, so * = r8*K+r9 */
    mov rax, r8
    imul rax, r14
    add rax, r9

    lea rax, [r15+4*rax]
    movss xmm1, [rax]
    addss xmm0, xmm1
    movss [rax], xmm0

    add r9, 1
    cmp r9, r14
    jnz smtm_KLoop

    add r8, 1
    cmp r8, r13
    jnz smtm_FLoop

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


ingestVector:
    /* rdi -> float *y -> rsi
       rsi -> float *buffer -> rdi
       rdx -> quad *rebuffer -> rdx
       rcx -> N -> r8
    */
    /* silly but more readable body by using sensible registers */
    mov r8, rcx
    xchg rsi, rdi

 
    xor rcx, rcx
1:
    mov r9, [rdx+8*rcx]
    mov eax, [rsi+4*r9]
    mov [rdi+4*rcx], eax

    add rcx, 1
    cmp rcx, r8
    jnz 1b

    ret


softmaxRowwise:
    /*  rdi -> N -> r12
        rsi -> K -> r13
        rdx -> float *X NxK -> r14
               start of row -> r15 
               rbx -> stash for max
               rbp -> stash for sum
    */
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rbx

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx

    xor r8, r8
softmaxRowwise_row:
  
    mov rax, r13
    imul rax, r8
    lea r15, [r14+4*rax]

    /* we'll use this later */
    pxor xmm0, xmm0
    movd rbp, xmm0

    /* compute max */
    xor rcx, rcx
1:  maxss xmm0, [r15+4*rcx] 
    add rcx, 1
    cmp rcx, r13
    jne 1b

    movd rbx, xmm0

    /* compute exp(v) */
    xor rcx, rcx
1:  

    movss xmm0, [r15+4*rcx]
    movd xmm1, rbx
    subss xmm0, xmm1
    /* preserve rcx, r8 */
    push r8
    push rcx
    call exp
    pop rcx
    pop r8
    movss [r15+4*rcx], xmm0
    movd xmm1, rbp
    addss xmm1, xmm0
    movd rbp, xmm1

    add rcx, 1
    cmp rcx, r13
    jne 1b

    /* normalize */
    movd xmm1, rbp
    xor rcx, rcx
1:  movss xmm0, [r15+4*rcx]
    divss xmm0, xmm1
    movss [r15+4*rcx], xmm0

    add rcx, 1
    cmp rcx, r13
    jne 1b

    add r8, 1
    cmp r8, r12
    jne softmaxRowwise_row

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret


mmul:
    /* 
       for i=1, ..., M
           Y(:,i) = X*W(:,i)

       slow, but... whatevs

       rdi -> N -> -8 
       rsi -> F -> -16
       rdx -> M -> -24
       rcx -> float *Y NxM -> -32
       r8 -> float *X NxF -> -40
       r9 -> float *W FxM -> -48
             N float storage -> r13 (rbp-56) START
             F float storage -> r14 (rbp-56-N)
             stack size      -> r15        

       0, ... F-1 in r12

       [ N          ] rbp-8
       [ F          ] rbp-16
       [ M          ] rbp-24
       [ float * Y  ] rbp-32
       [ float * X  ] rbp-40
       [ float * W  ] rbp-48
       [ float *b1  ] rbp-56
       [ float *b2  ] rbp-56-N

       Allocate N+F+48
    */
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rbp, rsp
    /* allocate stack */
    mov r15, rdi
    add r15, rsi
    shl r15, 2
    add r15, 448
    sub rsp, r15

    mov [rbp-8], rdi
    mov [rbp-16], rsi
    mov [rbp-24], rdx
    mov [rbp-32], rcx
    mov [rbp-40], r8
    mov [rbp-48], r9

    lea rdi, [rbp-56]
    /* calculate offsets -- remember, the place where they start on the stack is the
       end since we'll write increasing in memory like sane people */
    /* Nx4, Fx4 */
    mov rax, [rbp-8] 
    mov rcx, [rbp-16] 
    neg rax
    neg rcx
    lea r13, [rdi+4*rax]
    lea r14, [r13+4*rcx]

    xor r12, r12
mmul_colloop:
   
    /* transpose W(:,i) to a vector 
       *rsi -> *rdi  
       rdx is #per row*4
       rcx is #rows
     */
    mov rdi, r14
    mov rsi, [rbp-48]
    lea rsi, [rsi+4*r12]
    mov rdx, [rbp-24]
    shl rdx, 2
    mov rcx, [rbp-16]

    /* note the E! -- it's a float */
1:  mov eax, [rsi]
    mov [rdi], eax
    add rdi, 4
    add rsi, rdx
    sub rcx, 1
    jnz 1b


    /* now r14 points to the r12th column of W, set up a X*r14 call to write to r13 */
    mov rdi, [rbp-8]
    mov rsi, [rbp-16]
    mov rdx, r13
    mov rcx, [rbp-40]
    mov r8, r14
    call mmulv

    /* transpose to write to Y(:,i) 
        rsi -> rdi
        rdx is #per row*4
        rcx is #rows
     */
    mov rdi, [rbp-32]
    lea rdi, [rdi+4*r12]
    mov rsi, r13
    mov rcx, [rbp-8]
    mov rdx, [rbp-24]
    shl rdx, 2
1:  mov eax, [rsi]
    mov [rdi], eax
    add rsi, 4
    add rdi, rdx
    sub rcx, 1
    jnz 1b

    /* loop */
    add r12, 1
    cmp r12, [rbp-24]
    jnz mmul_colloop

    add rsp, r15 
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


mmulv:
    /* rdi -> N -> r12
       rsi -> F -> r13
       rdx -> float *y Nx1 -> r14
       rcx -> float *X NxF-> r15
       r8 -> float *w Fx1 -> rbx

       y = Xw
       
       i -> r11
    */
    push r12
    push r13
    push r14
    push r15
    push rbx

    /* shuffle them out */
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx
    mov rbx, r8

    /* compute the number of vectorized dps
       and unvectorized dps into r9, r10 */
    mov r10, r13
    shr r10, 2
    mov r9, r10
    shl r10, 2
    neg r10
    add r10, r13

    pxor xmm2, xmm2

    mov r11, r12
1:

    /* setup */
    mov rsi, rbx
    movaps xmm0, xmm2

    mov rcx, r9
    /* don't do vectorized code if we don't need to (will segfault) */
    cmp rcx, 0
    jz mmulv_dounvec
2:
    movups xmm1, [r15]
    movups xmm3, [rsi]
    mulps xmm1, xmm3
    addps xmm0, xmm1
    add rsi, 16
    add r15, 16
    sub rcx, 1
    jnz 2b

mmulv_dounvec:
    mov rcx, r10
    cmp rcx, 0
    jz 3f
2:
    movss xmm1, [r15]
    movss xmm3, [rsi]
    mulss xmm1, xmm3
    addss xmm0, xmm1
    add rsi, 4
    add r15, 4
    sub rcx, 1
    jnz 2b

3:
    /* reduce the 8 floats */
    haddps xmm0, xmm0
    haddps xmm0, xmm0

    movss [r14], xmm0

    add r14, 4
    sub r11, 1
    jnz 1b

    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12
    ret



isnumspecial:
    /* return whether the float xmm0 is
        ordinary: 0
        NaN: 1
        +Inf: 2
        -Inf: 3
    */
    xor rax, rax

    movd edx, xmm0
    mov ecx, edx
    and ecx, 0x7f800000
    cmp ecx, 0x7f800000
    /* ordinary, just exit */
    jnz 3f
    test edx, 0x7fffff
    jz 1f
    inc rax
    jmp 3f 
1:  mov al, 2
    bt edx, 31
    jnc 3f
    inc al
3:  ret 



dlogten:
    /* return how many decimal digits are in an int */
    mov rsi, 10
    xor rcx, rcx   
    mov rax, rdi
1:  xor rdx, rdx
    div rsi
    add rcx, 1
    cmp rax, 0
    jnz 1b
    mov rax, rcx
    ret

ofill:
    /* fill rdi with rsi floats of ones */
    mov rcx, rsi
    mov eax, 0x3f800000
    rep stosd
    ret

tenee:
    /* if we get called from c, only the low dword will be set */
    movss xmm0, [m_1]
    movss xmm1, [m_10]
    cmp edi, 0
    /* done if zero, diff branch if neg */
    je 3f
    jg 1f
    movss xmm1, [m_0p1]
    neg edi
1:  mulss xmm0, xmm1
    sub rdi, 1
    jnz 1b
3:  ret

tanh:
    call exp
    rcpss xmm1, xmm0
    /* 0 = e(x) , 1 = e(-x) */

    /* 2 = (e(x)-e(-x)) */
    movss xmm2, xmm0
    subss xmm2, xmm1

    /* 0 = e(x) + e(-x) */
    addss xmm0, xmm1
    /* 2 = (e(x)-e(-x)) / (e(x)+e(-x)) */
    divss xmm2, xmm0
    movss xmm0, xmm2

    ret

ln:
    /* compute ln(xmm0) */
    call log2
    movss xmm1, [m_log2e]
    divss xmm0, xmm1 
    ret

log2:
    /* test for special cases and propagate them through
      really we should do log(-inf) = nan but... */
    call isnumspecial
    cmp rax, 0
    jnz 3f

    /* compute log2(xmm0) */
    movd eax, xmm0

    /* strip off only fractional; exponent handled by shring */
    mov ecx, eax
    and ecx, 0x007fffff
    or ecx, 0x3f800000
    movd xmm0, ecx

    shr eax, 23
    sub eax, 127
    cvtsi2ss xmm1, eax

    /* compute xmm2 = t = (x-1)/(x+1) */
    movss xmm2, xmm0
    movss xmm3, xmm0
    subss xmm2, [m_1]
    addss xmm3, [m_1]
    divss xmm2, xmm3

    /* xmm0 = accum 
       xmm1 = terms from exponent
       xmm2 = t = (x-1)/(x+1)
            = t * (t^2)^i
       xmm3 = t*t     
       compute xmm0 = \sum_{i=0} t^(2i+1) / (2i+1) */

    /* ok set up */
    movss xmm3, xmm2
    mulss xmm3, xmm2
    pxor xmm0, xmm0
    xor rcx, rcx
1:
    /* rax = 2*rcx+1 -> xmm5 */
    lea rax, [2*rcx+1]
    cvtsi2ss xmm5, rax

    /* accum += t^n / (n) */
    movss xmm4, xmm2
    divss xmm4, xmm5
    addss xmm0, xmm4
    /* keep accumulating t^(n+2) = t^n * t^2 */
    mulss xmm2, xmm3

    add rcx, 1
    cmp rcx, 8
    jne 1b

    /* exponent + (2/ln2) \sum_{i=0} t^(2i+1) / (2i+1) */
    mulss xmm0, [m_2oln2]
    addss xmm0, xmm1

3:
    ret



exp:
    /* exp 
    https://math.stackexchange.com/questions/55830/how-to-calculate-ex-with-a-standard-calculator

    Basically, set al if it's negative
    Divide by two until we get to in [0,0.1], keep count in rcx
    Approximate by ((x+3)^2 + 3) / ((x-3)^2 + 3)
    Square rcx
    Reciprocate if al set
    Clobbers only rax, rcx */

    /* special test */
    call isnumspecial
    cmp rax, 0
    jnz 4f 

    xor ecx, ecx
    xor eax, eax

    comiss xmm0, [m_0]
    ja 1f
    /* flip sign, set al to 1 so we rcp at end */
    negatexmm xmm0, eax
    mov al, 1
1:

    /* if it's already < 0.1, just approx */
    comiss xmm0, [m_0p1]
    jle 2f
    /* else, divide by two */
    movss xmm1, [m_0p1]
    movss xmm2, [m_0p5]
1:
    inc rcx
    mulss xmm0, xmm2
    comiss xmm0, xmm1
    ja 1b

2:
    movss xmm2, [m_3]
    /* xmm0 \in [0,0.1], al is sign, and rcx is # of powers we knocked off */   
    /* ((x+3)*(x+3) + 3) / ((x-3)*(x-3) + 3); numerator: xmm0; denom: xmm1 */
    movss xmm1, xmm0

    addss xmm0, xmm2
    mulss xmm0, xmm0
    addss xmm0, xmm2

    subss xmm1, xmm2
    mulss xmm1, xmm1
    addss xmm1, xmm2

    divss xmm0, xmm1

    /* do we have to square? */
    cmp rcx, 0
    je 3f

1:
    mulss xmm0, xmm0
    dec rcx
    jnz 1b

3:
    /* do we have to reciprocate since the arg was negative? */
    cmp al, 0
    je 4f
    rcpss xmm0, xmm0
4:
    ret


