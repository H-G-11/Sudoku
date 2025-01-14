��-
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.1-0-g85c8b2a817f8��&
�
conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *!
shared_nameconv2d_44/kernel
}
$conv2d_44/kernel/Read/ReadVariableOpReadVariableOpconv2d_44/kernel*&
_output_shapes
:	 *
dtype0
t
conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_44/bias
m
"conv2d_44/bias/Read/ReadVariableOpReadVariableOpconv2d_44/bias*
_output_shapes
: *
dtype0
�
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *!
shared_nameconv2d_45/kernel
}
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*&
_output_shapes
:		 *
dtype0
t
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_45/bias
m
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes
: *
dtype0
�
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
:		 *
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
: *
dtype0
�
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:			 *!
shared_nameconv2d_47/kernel
}
$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*&
_output_shapes
:			 *
dtype0
t
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_47/bias
m
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes
: *
dtype0
�
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_21/gamma
�
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_21/beta
�
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_21/moving_mean
�
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_21/moving_variance
�
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		�@*!
shared_nameconv2d_48/kernel
~
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*'
_output_shapes
:		�@*
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_22/gamma
�
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_22/beta
�
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_22/moving_mean
�
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_22/moving_variance
�
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_49/kernel
}
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_49/bias
m
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_23/gamma
�
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_23/beta
�
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_23/moving_mean
�
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_23/moving_variance
�
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_50/kernel
}
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_24/gamma
�
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_24/beta
�
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_24/moving_mean
�
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_24/moving_variance
�
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
:@ *
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
: *
dtype0
�
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_25/gamma
�
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_25/beta
�
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_25/moving_mean
�
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_25/moving_variance
�
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_52/kernel
}
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_52/bias
m
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes
: *
dtype0
�
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_26/gamma
�
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_26/beta
�
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_26/moving_mean
�
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_26/moving_variance
�
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_53/kernel
}
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_53/bias
m
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes
: *
dtype0
�
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_27/gamma
�
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_27/beta
�
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_27/moving_mean
�
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_27/moving_variance
�
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
: *
dtype0
�
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_28/gamma
�
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_28/beta
�
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_28/moving_mean
�
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_28/moving_variance
�
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:)	*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
:)	*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_nameAdam/conv2d_44/kernel/m
�
+Adam/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/m*&
_output_shapes
:	 *
dtype0
�
Adam/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_44/bias/m
{
)Adam/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *(
shared_nameAdam/conv2d_45/kernel/m
�
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*&
_output_shapes
:		 *
dtype0
�
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_45/bias/m
{
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *(
shared_nameAdam/conv2d_46/kernel/m
�
+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*&
_output_shapes
:		 *
dtype0
�
Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/m
{
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:			 *(
shared_nameAdam/conv2d_47/kernel/m
�
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*&
_output_shapes
:			 *
dtype0
�
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/m
{
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_21/gamma/m
�
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_21/beta/m
�
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		�@*(
shared_nameAdam/conv2d_48/kernel/m
�
+Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/m*'
_output_shapes
:		�@*
dtype0
�
Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_48/bias/m
{
)Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_22/gamma/m
�
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_22/beta/m
�
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_49/kernel/m
�
+Adam/conv2d_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_49/bias/m
{
)Adam/conv2d_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_23/gamma/m
�
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_23/beta/m
�
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_50/kernel/m
�
+Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_50/bias/m
{
)Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_24/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_24/gamma/m
�
7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_24/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_24/beta/m
�
6Adam/batch_normalization_24/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_51/kernel/m
�
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*&
_output_shapes
:@ *
dtype0
�
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/m
{
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_25/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_25/gamma/m
�
7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_25/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_25/beta/m
�
6Adam/batch_normalization_25/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_52/kernel/m
�
+Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_52/bias/m
{
)Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_26/gamma/m
�
7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_26/beta/m
�
6Adam/batch_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_53/kernel/m
�
+Adam/conv2d_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_53/bias/m
{
)Adam/conv2d_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_27/gamma/m
�
7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_27/beta/m
�
6Adam/batch_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_54/kernel/m
�
+Adam/conv2d_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/m
{
)Adam/conv2d_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/m*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_28/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_28/gamma/m
�
7Adam/batch_normalization_28/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_28/gamma/m*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_28/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_28/beta/m
�
6Adam/batch_normalization_28/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_28/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:)	*(
shared_nameAdam/conv2d_55/kernel/m
�
+Adam/conv2d_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/m*&
_output_shapes
:)	*
dtype0
�
Adam/conv2d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/conv2d_55/bias/m
{
)Adam/conv2d_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/m*
_output_shapes
:	*
dtype0
�
Adam/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_nameAdam/conv2d_44/kernel/v
�
+Adam/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/v*&
_output_shapes
:	 *
dtype0
�
Adam/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_44/bias/v
{
)Adam/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *(
shared_nameAdam/conv2d_45/kernel/v
�
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*&
_output_shapes
:		 *
dtype0
�
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_45/bias/v
{
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *(
shared_nameAdam/conv2d_46/kernel/v
�
+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*&
_output_shapes
:		 *
dtype0
�
Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/v
{
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:			 *(
shared_nameAdam/conv2d_47/kernel/v
�
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*&
_output_shapes
:			 *
dtype0
�
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/v
{
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_21/gamma/v
�
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_21/beta/v
�
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		�@*(
shared_nameAdam/conv2d_48/kernel/v
�
+Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/v*'
_output_shapes
:		�@*
dtype0
�
Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_48/bias/v
{
)Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_22/gamma/v
�
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_22/beta/v
�
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_49/kernel/v
�
+Adam/conv2d_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_49/bias/v
{
)Adam/conv2d_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_23/gamma/v
�
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_23/beta/v
�
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_50/kernel/v
�
+Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_50/bias/v
{
)Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_24/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_24/gamma/v
�
7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_24/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_24/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_24/beta/v
�
6Adam/batch_normalization_24/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_24/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_51/kernel/v
�
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*&
_output_shapes
:@ *
dtype0
�
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/v
{
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_25/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_25/gamma/v
�
7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_25/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_25/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_25/beta/v
�
6Adam/batch_normalization_25/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_25/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_52/kernel/v
�
+Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_52/bias/v
{
)Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_26/gamma/v
�
7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_26/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_26/beta/v
�
6Adam/batch_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_26/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_53/kernel/v
�
+Adam/conv2d_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_53/bias/v
{
)Adam/conv2d_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_27/gamma/v
�
7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_27/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_27/beta/v
�
6Adam/batch_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_27/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_54/kernel/v
�
+Adam/conv2d_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/v
{
)Adam/conv2d_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/v*
_output_shapes
: *
dtype0
�
#Adam/batch_normalization_28/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_28/gamma/v
�
7Adam/batch_normalization_28/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_28/gamma/v*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_28/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_28/beta/v
�
6Adam/batch_normalization_28/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_28/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:)	*(
shared_nameAdam/conv2d_55/kernel/v
�
+Adam/conv2d_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/v*&
_output_shapes
:)	*
dtype0
�
Adam/conv2d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/conv2d_55/bias/v
{
)Adam/conv2d_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
layer_with_weights-18
layer-20
layer-21
layer_with_weights-19
layer-22
layer-23
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
h

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
�
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^trainable_variables
_regularization_losses
`	variables
a	keras_api
h

bkernel
cbias
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
�
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
h

qkernel
rbias
strainable_variables
tregularization_losses
u	variables
v	keras_api
�
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem� m�%m�&m�+m�,m�1m�2m�<m�=m�Dm�Em�Km�Lm�Sm�Tm�Zm�[m�bm�cm�im�jm�qm�rm�xm�ym�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v� v�%v�&v�+v�,v�1v�2v�<v�=v�Dv�Ev�Kv�Lv�Sv�Tv�Zv�[v�bv�cv�iv�jv�qv�rv�xv�yv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
�
0
 1
%2
&3
+4
,5
16
27
<8
=9
D10
E11
K12
L13
S14
T15
Z16
[17
b18
c19
i20
j21
q22
r23
x24
y25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
 
�
0
 1
%2
&3
+4
,5
16
27
<8
=9
>10
?11
D12
E13
K14
L15
M16
N17
S18
T19
Z20
[21
\22
]23
b24
c25
i26
j27
k28
l29
q30
r31
x32
y33
z34
{35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�
trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
 
\Z
VARIABLE_VALUEconv2d_44/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_44/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
�
!trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
"regularization_losses
�metrics
#	variables
\Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
�
'trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
(regularization_losses
�metrics
)	variables
\Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
�
-trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
.regularization_losses
�metrics
/	variables
\Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
�
3trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
4regularization_losses
�metrics
5	variables
 
 
 
�
7trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
8regularization_losses
�metrics
9	variables
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
>2
?3
�
@trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Aregularization_losses
�metrics
B	variables
\Z
VARIABLE_VALUEconv2d_48/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_48/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
�
Ftrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Gregularization_losses
�metrics
H	variables
 
ge
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
M2
N3
�
Otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Pregularization_losses
�metrics
Q	variables
\Z
VARIABLE_VALUEconv2d_49/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_49/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
�
Utrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Vregularization_losses
�metrics
W	variables
 
ge
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
 

Z0
[1
\2
]3
�
^trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
_regularization_losses
�metrics
`	variables
\Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
�
dtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
eregularization_losses
�metrics
f	variables
 
hf
VARIABLE_VALUEbatch_normalization_24/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_24/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_24/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_24/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
k2
l3
�
mtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
nregularization_losses
�metrics
o	variables
][
VARIABLE_VALUEconv2d_51/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_51/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
 

q0
r1
�
strainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
tregularization_losses
�metrics
u	variables
 
hf
VARIABLE_VALUEbatch_normalization_25/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_25/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_25/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_25/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
z2
{3
�
|trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
}regularization_losses
�metrics
~	variables
][
VARIABLE_VALUEconv2d_52/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_52/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
 
hf
VARIABLE_VALUEbatch_normalization_26/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_26/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_26/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_26/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
][
VARIABLE_VALUEconv2d_53/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_53/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
 
hf
VARIABLE_VALUEbatch_normalization_27/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_27/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_27/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_27/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
][
VARIABLE_VALUEconv2d_54/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_54/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
 
hf
VARIABLE_VALUEbatch_normalization_28/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_28/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_28/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_28/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
 
 
 
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
][
VARIABLE_VALUEconv2d_55/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_55/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
 
 
 
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 
|
>0
?1
M2
N3
\4
]5
k6
l7
z8
{9
�10
�11
�12
�13
�14
�15

�0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 

\0
]1
 
 
 
 
 
 
 
 
 

k0
l1
 
 
 
 
 
 
 
 
 

z0
{1
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
}
VARIABLE_VALUEAdam/conv2d_44/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_48/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_24/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_51/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_51/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_25/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_52/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_52/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_26/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_53/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_53/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_27/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_54/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_54/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_28/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_28/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_55/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_55/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_44/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_48/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_24/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_24/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_51/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_51/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_25/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_25/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_52/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_52/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_26/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_26/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_53/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_53/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_27/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_27/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_54/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_54/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_28/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_28/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv2d_55/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_55/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_5Placeholder*/
_output_shapes
:���������			*
dtype0*$
shape:���������			
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5conv2d_44/kernelconv2d_44/biasconv2d_45/kernelconv2d_45/biasconv2d_46/kernelconv2d_46/biasconv2d_47/kernelconv2d_47/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_49/kernelconv2d_49/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_50/kernelconv2d_50/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_51/kernelconv2d_51/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_52/kernelconv2d_52/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_53/kernelconv2d_53/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_54/kernelconv2d_54/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_55/kernelconv2d_55/bias*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_242740
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�8
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_44/kernel/Read/ReadVariableOp"conv2d_44/bias/Read/ReadVariableOp$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_44/kernel/m/Read/ReadVariableOp)Adam/conv2d_44/bias/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp+Adam/conv2d_48/kernel/m/Read/ReadVariableOp)Adam/conv2d_48/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp+Adam/conv2d_49/kernel/m/Read/ReadVariableOp)Adam/conv2d_49/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp+Adam/conv2d_50/kernel/m/Read/ReadVariableOp)Adam/conv2d_50/bias/m/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_24/beta/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_25/beta/m/Read/ReadVariableOp+Adam/conv2d_52/kernel/m/Read/ReadVariableOp)Adam/conv2d_52/bias/m/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_26/beta/m/Read/ReadVariableOp+Adam/conv2d_53/kernel/m/Read/ReadVariableOp)Adam/conv2d_53/bias/m/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_27/beta/m/Read/ReadVariableOp+Adam/conv2d_54/kernel/m/Read/ReadVariableOp)Adam/conv2d_54/bias/m/Read/ReadVariableOp7Adam/batch_normalization_28/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_28/beta/m/Read/ReadVariableOp+Adam/conv2d_55/kernel/m/Read/ReadVariableOp)Adam/conv2d_55/bias/m/Read/ReadVariableOp+Adam/conv2d_44/kernel/v/Read/ReadVariableOp)Adam/conv2d_44/bias/v/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp+Adam/conv2d_48/kernel/v/Read/ReadVariableOp)Adam/conv2d_48/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp+Adam/conv2d_49/kernel/v/Read/ReadVariableOp)Adam/conv2d_49/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp+Adam/conv2d_50/kernel/v/Read/ReadVariableOp)Adam/conv2d_50/bias/v/Read/ReadVariableOp7Adam/batch_normalization_24/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_24/beta/v/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp7Adam/batch_normalization_25/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_25/beta/v/Read/ReadVariableOp+Adam/conv2d_52/kernel/v/Read/ReadVariableOp)Adam/conv2d_52/bias/v/Read/ReadVariableOp7Adam/batch_normalization_26/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_26/beta/v/Read/ReadVariableOp+Adam/conv2d_53/kernel/v/Read/ReadVariableOp)Adam/conv2d_53/bias/v/Read/ReadVariableOp7Adam/batch_normalization_27/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_27/beta/v/Read/ReadVariableOp+Adam/conv2d_54/kernel/v/Read/ReadVariableOp)Adam/conv2d_54/bias/v/Read/ReadVariableOp7Adam/batch_normalization_28/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_28/beta/v/Read/ReadVariableOp+Adam/conv2d_55/kernel/v/Read/ReadVariableOp)Adam/conv2d_55/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_245171
�!
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_44/kernelconv2d_44/biasconv2d_45/kernelconv2d_45/biasconv2d_46/kernelconv2d_46/biasconv2d_47/kernelconv2d_47/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_49/kernelconv2d_49/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_50/kernelconv2d_50/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_51/kernelconv2d_51/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_52/kernelconv2d_52/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_53/kernelconv2d_53/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_54/kernelconv2d_54/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_55/kernelconv2d_55/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_44/kernel/mAdam/conv2d_44/bias/mAdam/conv2d_45/kernel/mAdam/conv2d_45/bias/mAdam/conv2d_46/kernel/mAdam/conv2d_46/bias/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/m#Adam/batch_normalization_21/gamma/m"Adam/batch_normalization_21/beta/mAdam/conv2d_48/kernel/mAdam/conv2d_48/bias/m#Adam/batch_normalization_22/gamma/m"Adam/batch_normalization_22/beta/mAdam/conv2d_49/kernel/mAdam/conv2d_49/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/mAdam/conv2d_50/kernel/mAdam/conv2d_50/bias/m#Adam/batch_normalization_24/gamma/m"Adam/batch_normalization_24/beta/mAdam/conv2d_51/kernel/mAdam/conv2d_51/bias/m#Adam/batch_normalization_25/gamma/m"Adam/batch_normalization_25/beta/mAdam/conv2d_52/kernel/mAdam/conv2d_52/bias/m#Adam/batch_normalization_26/gamma/m"Adam/batch_normalization_26/beta/mAdam/conv2d_53/kernel/mAdam/conv2d_53/bias/m#Adam/batch_normalization_27/gamma/m"Adam/batch_normalization_27/beta/mAdam/conv2d_54/kernel/mAdam/conv2d_54/bias/m#Adam/batch_normalization_28/gamma/m"Adam/batch_normalization_28/beta/mAdam/conv2d_55/kernel/mAdam/conv2d_55/bias/mAdam/conv2d_44/kernel/vAdam/conv2d_44/bias/vAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/vAdam/conv2d_46/kernel/vAdam/conv2d_46/bias/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/v#Adam/batch_normalization_21/gamma/v"Adam/batch_normalization_21/beta/vAdam/conv2d_48/kernel/vAdam/conv2d_48/bias/v#Adam/batch_normalization_22/gamma/v"Adam/batch_normalization_22/beta/vAdam/conv2d_49/kernel/vAdam/conv2d_49/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/vAdam/conv2d_50/kernel/vAdam/conv2d_50/bias/v#Adam/batch_normalization_24/gamma/v"Adam/batch_normalization_24/beta/vAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/v#Adam/batch_normalization_25/gamma/v"Adam/batch_normalization_25/beta/vAdam/conv2d_52/kernel/vAdam/conv2d_52/bias/v#Adam/batch_normalization_26/gamma/v"Adam/batch_normalization_26/beta/vAdam/conv2d_53/kernel/vAdam/conv2d_53/bias/v#Adam/batch_normalization_27/gamma/v"Adam/batch_normalization_27/beta/vAdam/conv2d_54/kernel/vAdam/conv2d_54/bias/v#Adam/batch_normalization_28/gamma/v"Adam/batch_normalization_28/beta/vAdam/conv2d_55/kernel/vAdam/conv2d_55/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_245610��!
�

�
E__inference_conv2d_51_layer_call_and_return_conditional_losses_244090

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_241845

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_44_layer_call_and_return_conditional_losses_241009

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_240848

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_54_layer_call_and_return_conditional_losses_241810

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_24_layer_call_fn_244002

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_2405362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_21_layer_call_fn_243635

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_2411632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244563

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_53_layer_call_and_return_conditional_losses_241710

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_241763

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
v
J__inference_concatenate_10_layer_call_and_return_conditional_losses_244678
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������		)2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������		)2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������		 :���������			:Y U
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������			
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_240775

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_240983

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_25_layer_call_fn_244163

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_2406712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_23_layer_call_fn_243931

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2404632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

*__inference_conv2d_52_layer_call_fn_244247

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2416102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_241263

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_241645

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_240952

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
E
.__inference_softmax_map_4_layer_call_fn_244719
x
identity�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_2419522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������			:R N
/
_output_shapes
:���������			

_user_specified_namex
�
�
7__inference_batch_normalization_26_layer_call_fn_244375

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_2416632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244053

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_55_layer_call_and_return_conditional_losses_244694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:)	*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		)::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		)
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_25_layer_call_fn_244214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_2415452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_242357
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*J
_read_only_resource_inputs,
*(	
 !"%&'(+,-.123478*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2422422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_240255

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243989

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_240567

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_51_layer_call_and_return_conditional_losses_241510

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
v
.__inference_concatenate_9_layer_call_fn_243507
inputs_0
inputs_1
inputs_2
inputs_3
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_9_layer_call_and_return_conditional_losses_2411152
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*
_input_shapesn
l:���������		 :���������		 :���������		 :���������		 :Y U
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/3
�
�
7__inference_batch_normalization_26_layer_call_fn_244362

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_2416452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_28_layer_call_fn_244671

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2409832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_54_layer_call_and_return_conditional_losses_244534

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

*__inference_conv2d_49_layer_call_fn_243803

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_2413102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_45_layer_call_and_return_conditional_losses_243441

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�

�
E__inference_conv2d_47_layer_call_and_return_conditional_losses_243481

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:			 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_23_layer_call_fn_243854

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2413452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_24_layer_call_fn_244079

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_2414632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_240671

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_23_layer_call_fn_243867

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2413632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_23_layer_call_fn_243918

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2404322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244433

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_26_layer_call_fn_244298

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_2407442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_27_layer_call_fn_244510

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2417452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_55_layer_call_and_return_conditional_losses_241925

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:)	*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		)::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		)
 
_user_specified_nameinputs
�

*__inference_conv2d_45_layer_call_fn_243450

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_2410362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_240536

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_241363

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

*__inference_conv2d_50_layer_call_fn_243951

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_2414102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_44_layer_call_and_return_conditional_losses_243421

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
��
�+
C__inference_model_4_layer_call_and_return_conditional_losses_243176

inputs,
(conv2d_44_conv2d_readvariableop_resource-
)conv2d_44_biasadd_readvariableop_resource,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceC
?batch_normalization_22_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource
identity��6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1�6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1�6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1�6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1�6batch_normalization_28/FusedBatchNormV3/ReadVariableOp�8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_28/ReadVariableOp�'batch_normalization_28/ReadVariableOp_1� conv2d_44/BiasAdd/ReadVariableOp�conv2d_44/Conv2D/ReadVariableOp� conv2d_45/BiasAdd/ReadVariableOp�conv2d_45/Conv2D/ReadVariableOp� conv2d_46/BiasAdd/ReadVariableOp�conv2d_46/Conv2D/ReadVariableOp� conv2d_47/BiasAdd/ReadVariableOp�conv2d_47/Conv2D/ReadVariableOp� conv2d_48/BiasAdd/ReadVariableOp�conv2d_48/Conv2D/ReadVariableOp� conv2d_49/BiasAdd/ReadVariableOp�conv2d_49/Conv2D/ReadVariableOp� conv2d_50/BiasAdd/ReadVariableOp�conv2d_50/Conv2D/ReadVariableOp� conv2d_51/BiasAdd/ReadVariableOp�conv2d_51/Conv2D/ReadVariableOp� conv2d_52/BiasAdd/ReadVariableOp�conv2d_52/Conv2D/ReadVariableOp� conv2d_53/BiasAdd/ReadVariableOp�conv2d_53/Conv2D/ReadVariableOp� conv2d_54/BiasAdd/ReadVariableOp�conv2d_54/Conv2D/ReadVariableOp� conv2d_55/BiasAdd/ReadVariableOp�conv2d_55/Conv2D/ReadVariableOp�
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype02!
conv2d_44/Conv2D/ReadVariableOp�
conv2d_44/Conv2DConv2Dinputs'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_44/Conv2D�
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_44/BiasAdd/ReadVariableOp�
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_44/BiasAdd~
conv2d_44/TanhTanhconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_44/Tanh�
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02!
conv2d_45/Conv2D/ReadVariableOp�
conv2d_45/Conv2DConv2Dinputs'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_45/Conv2D�
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_45/BiasAdd/ReadVariableOp�
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_45/BiasAdd~
conv2d_45/TanhTanhconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_45/Tanh�
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02!
conv2d_46/Conv2D/ReadVariableOp�
conv2d_46/Conv2DConv2Dinputs'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_46/Conv2D�
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp�
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_46/BiasAdd~
conv2d_46/TanhTanhconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_46/Tanh�
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:			 *
dtype02!
conv2d_47/Conv2D/ReadVariableOp�
conv2d_47/Conv2DConv2Dinputs'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_47/Conv2D�
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOp�
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_47/BiasAdd~
conv2d_47/TanhTanhconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_47/Tanhx
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axis�
concatenate_9/concatConcatV2conv2d_44/Tanh:y:0conv2d_45/Tanh:y:0conv2d_46/Tanh:y:0conv2d_47/Tanh:y:0"concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������		�2
concatenate_9/concat�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_21/ReadVariableOp�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_21/ReadVariableOp_1�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3concatenate_9/concat:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_21/FusedBatchNormV3�
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*'
_output_shapes
:		�@*
dtype02!
conv2d_48/Conv2D/ReadVariableOp�
conv2d_48/Conv2DConv2D+batch_normalization_21/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_48/Conv2D�
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp�
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_48/BiasAdd~
conv2d_48/TanhTanhconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_48/Tanh�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_22/ReadVariableOp�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_22/ReadVariableOp_1�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_48/Tanh:y:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_22/FusedBatchNormV3�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_49/BiasAdd~
conv2d_49/TanhTanhconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_49/Tanh�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_23/ReadVariableOp�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_23/ReadVariableOp_1�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_49/Tanh:y:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_23/FusedBatchNormV3�
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_50/Conv2D/ReadVariableOp�
conv2d_50/Conv2DConv2D+batch_normalization_23/FusedBatchNormV3:y:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_50/Conv2D�
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp�
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_50/Relu�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_50/Relu:activations:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_51/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_51/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3�
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_52/Relu�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_52/Relu:activations:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_53/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_53/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3�
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_54/Conv2D/ReadVariableOp�
conv2d_54/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_54/Conv2D�
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp�
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_54/Relu�
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp�
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1�
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_54/Relu:activations:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3z
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axis�
concatenate_10/concatConcatV2+batch_normalization_28/FusedBatchNormV3:y:0inputs#concatenate_10/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������		)2
concatenate_10/concat�
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:)	*
dtype02!
conv2d_55/Conv2D/ReadVariableOp�
conv2d_55/Conv2DConv2Dconcatenate_10/concat:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			*
paddingSAME*
strides
2
conv2d_55/Conv2D�
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp�
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			2
conv2d_55/BiasAdd�
#softmax_map_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#softmax_map_4/Max/reduction_indices�
softmax_map_4/MaxMaxconv2d_55/BiasAdd:output:0,softmax_map_4/Max/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
softmax_map_4/Max�
softmax_map_4/subSubconv2d_55/BiasAdd:output:0softmax_map_4/Max:output:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/sub~
softmax_map_4/ExpExpsoftmax_map_4/sub:z:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/Exp�
#softmax_map_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#softmax_map_4/Sum/reduction_indices�
softmax_map_4/SumSumsoftmax_map_4/Exp:y:0,softmax_map_4/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
softmax_map_4/Sum�
softmax_map_4/truedivRealDivsoftmax_map_4/Exp:y:0softmax_map_4/Sum:output:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/truediv�
IdentityIdentitysoftmax_map_4/truediv:z:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_21_layer_call_fn_243622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_2411452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�

�
E__inference_conv2d_52_layer_call_and_return_conditional_losses_244238

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_242613
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2424982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244645

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

*__inference_conv2d_55_layer_call_fn_244703

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_2419252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		)::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		)
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_241745

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
Յ
�
C__inference_model_4_layer_call_and_return_conditional_losses_241961
input_5
conv2d_44_241020
conv2d_44_241022
conv2d_45_241047
conv2d_45_241049
conv2d_46_241074
conv2d_46_241076
conv2d_47_241101
conv2d_47_241103!
batch_normalization_21_241190!
batch_normalization_21_241192!
batch_normalization_21_241194!
batch_normalization_21_241196
conv2d_48_241221
conv2d_48_241223!
batch_normalization_22_241290!
batch_normalization_22_241292!
batch_normalization_22_241294!
batch_normalization_22_241296
conv2d_49_241321
conv2d_49_241323!
batch_normalization_23_241390!
batch_normalization_23_241392!
batch_normalization_23_241394!
batch_normalization_23_241396
conv2d_50_241421
conv2d_50_241423!
batch_normalization_24_241490!
batch_normalization_24_241492!
batch_normalization_24_241494!
batch_normalization_24_241496
conv2d_51_241521
conv2d_51_241523!
batch_normalization_25_241590!
batch_normalization_25_241592!
batch_normalization_25_241594!
batch_normalization_25_241596
conv2d_52_241621
conv2d_52_241623!
batch_normalization_26_241690!
batch_normalization_26_241692!
batch_normalization_26_241694!
batch_normalization_26_241696
conv2d_53_241721
conv2d_53_241723!
batch_normalization_27_241790!
batch_normalization_27_241792!
batch_normalization_27_241794!
batch_normalization_27_241796
conv2d_54_241821
conv2d_54_241823!
batch_normalization_28_241890!
batch_normalization_28_241892!
batch_normalization_28_241894!
batch_normalization_28_241896
conv2d_55_241936
conv2d_55_241938
identity��.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall�.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall�!conv2d_44/StatefulPartitionedCall�!conv2d_45/StatefulPartitionedCall�!conv2d_46/StatefulPartitionedCall�!conv2d_47/StatefulPartitionedCall�!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall�
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_44_241020conv2d_44_241022*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_2410092#
!conv2d_44/StatefulPartitionedCall�
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_45_241047conv2d_45_241049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_2410362#
!conv2d_45/StatefulPartitionedCall�
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_46_241074conv2d_46_241076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_2410632#
!conv2d_46/StatefulPartitionedCall�
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_47_241101conv2d_47_241103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_2410902#
!conv2d_47/StatefulPartitionedCall�
concatenate_9/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*conv2d_45/StatefulPartitionedCall:output:0*conv2d_46/StatefulPartitionedCall:output:0*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_9_layer_call_and_return_conditional_losses_2411152
concatenate_9/PartitionedCall�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0batch_normalization_21_241190batch_normalization_21_241192batch_normalization_21_241194batch_normalization_21_241196*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_24114520
.batch_normalization_21/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0conv2d_48_241221conv2d_48_241223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_48_layer_call_and_return_conditional_losses_2412102#
!conv2d_48/StatefulPartitionedCall�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_22_241290batch_normalization_22_241292batch_normalization_22_241294batch_normalization_22_241296*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_24124520
.batch_normalization_22/StatefulPartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_49_241321conv2d_49_241323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_2413102#
!conv2d_49/StatefulPartitionedCall�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_23_241390batch_normalization_23_241392batch_normalization_23_241394batch_normalization_23_241396*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_24134520
.batch_normalization_23/StatefulPartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_50_241421conv2d_50_241423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_2414102#
!conv2d_50/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_24_241490batch_normalization_24_241492batch_normalization_24_241494batch_normalization_24_241496*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_24144520
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_51_241521conv2d_51_241523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2415102#
!conv2d_51/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_25_241590batch_normalization_25_241592batch_normalization_25_241594batch_normalization_25_241596*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_24154520
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_52_241621conv2d_52_241623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2416102#
!conv2d_52/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_26_241690batch_normalization_26_241692batch_normalization_26_241694batch_normalization_26_241696*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_24164520
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_53_241721conv2d_53_241723*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2417102#
!conv2d_53/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_27_241790batch_normalization_27_241792batch_normalization_27_241794batch_normalization_27_241796*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_24174520
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_54_241821conv2d_54_241823*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_2418102#
!conv2d_54/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_28_241890batch_normalization_28_241892batch_normalization_28_241894batch_normalization_28_241896*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_24184520
.batch_normalization_28/StatefulPartitionedCall�
concatenate_10/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0input_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_10_layer_call_and_return_conditional_losses_2419062 
concatenate_10/PartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_55_241936conv2d_55_241938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_2419252#
!conv2d_55/StatefulPartitionedCall�
softmax_map_4/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_2419522
softmax_map_4/PartitionedCall�
IdentityIdentity&softmax_map_4/PartitionedCall:output:0/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�

�
E__inference_conv2d_49_layer_call_and_return_conditional_losses_241310

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243527

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_241463

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_241245

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_241863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243823

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_240224

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_24_layer_call_fn_244015

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_2405672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_48_layer_call_and_return_conditional_losses_241210

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:		�@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������		�::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
��
�0
C__inference_model_4_layer_call_and_return_conditional_losses_242966

inputs,
(conv2d_44_conv2d_readvariableop_resource-
)conv2d_44_biasadd_readvariableop_resource,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceC
?batch_normalization_22_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource2
.batch_normalization_26_readvariableop_resource4
0batch_normalization_26_readvariableop_1_resourceC
?batch_normalization_26_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_28_readvariableop_resource4
0batch_normalization_28_readvariableop_1_resourceC
?batch_normalization_28_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource
identity��%batch_normalization_21/AssignNewValue�'batch_normalization_21/AssignNewValue_1�6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1�%batch_normalization_22/AssignNewValue�'batch_normalization_22/AssignNewValue_1�6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1�%batch_normalization_23/AssignNewValue�'batch_normalization_23/AssignNewValue_1�6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1�%batch_normalization_24/AssignNewValue�'batch_normalization_24/AssignNewValue_1�6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_24/ReadVariableOp�'batch_normalization_24/ReadVariableOp_1�%batch_normalization_25/AssignNewValue�'batch_normalization_25/AssignNewValue_1�6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_25/ReadVariableOp�'batch_normalization_25/ReadVariableOp_1�%batch_normalization_26/AssignNewValue�'batch_normalization_26/AssignNewValue_1�6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_26/ReadVariableOp�'batch_normalization_26/ReadVariableOp_1�%batch_normalization_27/AssignNewValue�'batch_normalization_27/AssignNewValue_1�6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_27/ReadVariableOp�'batch_normalization_27/ReadVariableOp_1�%batch_normalization_28/AssignNewValue�'batch_normalization_28/AssignNewValue_1�6batch_normalization_28/FusedBatchNormV3/ReadVariableOp�8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_28/ReadVariableOp�'batch_normalization_28/ReadVariableOp_1� conv2d_44/BiasAdd/ReadVariableOp�conv2d_44/Conv2D/ReadVariableOp� conv2d_45/BiasAdd/ReadVariableOp�conv2d_45/Conv2D/ReadVariableOp� conv2d_46/BiasAdd/ReadVariableOp�conv2d_46/Conv2D/ReadVariableOp� conv2d_47/BiasAdd/ReadVariableOp�conv2d_47/Conv2D/ReadVariableOp� conv2d_48/BiasAdd/ReadVariableOp�conv2d_48/Conv2D/ReadVariableOp� conv2d_49/BiasAdd/ReadVariableOp�conv2d_49/Conv2D/ReadVariableOp� conv2d_50/BiasAdd/ReadVariableOp�conv2d_50/Conv2D/ReadVariableOp� conv2d_51/BiasAdd/ReadVariableOp�conv2d_51/Conv2D/ReadVariableOp� conv2d_52/BiasAdd/ReadVariableOp�conv2d_52/Conv2D/ReadVariableOp� conv2d_53/BiasAdd/ReadVariableOp�conv2d_53/Conv2D/ReadVariableOp� conv2d_54/BiasAdd/ReadVariableOp�conv2d_54/Conv2D/ReadVariableOp� conv2d_55/BiasAdd/ReadVariableOp�conv2d_55/Conv2D/ReadVariableOp�
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype02!
conv2d_44/Conv2D/ReadVariableOp�
conv2d_44/Conv2DConv2Dinputs'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_44/Conv2D�
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_44/BiasAdd/ReadVariableOp�
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_44/BiasAdd~
conv2d_44/TanhTanhconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_44/Tanh�
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02!
conv2d_45/Conv2D/ReadVariableOp�
conv2d_45/Conv2DConv2Dinputs'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_45/Conv2D�
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_45/BiasAdd/ReadVariableOp�
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_45/BiasAdd~
conv2d_45/TanhTanhconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_45/Tanh�
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02!
conv2d_46/Conv2D/ReadVariableOp�
conv2d_46/Conv2DConv2Dinputs'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_46/Conv2D�
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp�
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_46/BiasAdd~
conv2d_46/TanhTanhconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_46/Tanh�
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:			 *
dtype02!
conv2d_47/Conv2D/ReadVariableOp�
conv2d_47/Conv2DConv2Dinputs'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_47/Conv2D�
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOp�
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_47/BiasAdd~
conv2d_47/TanhTanhconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_47/Tanhx
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axis�
concatenate_9/concatConcatV2conv2d_44/Tanh:y:0conv2d_45/Tanh:y:0conv2d_46/Tanh:y:0conv2d_47/Tanh:y:0"concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������		�2
concatenate_9/concat�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_21/ReadVariableOp�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_21/ReadVariableOp_1�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3concatenate_9/concat:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_21/FusedBatchNormV3�
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_21/AssignNewValue�
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_21/AssignNewValue_1�
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*'
_output_shapes
:		�@*
dtype02!
conv2d_48/Conv2D/ReadVariableOp�
conv2d_48/Conv2DConv2D+batch_normalization_21/FusedBatchNormV3:y:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_48/Conv2D�
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp�
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_48/BiasAdd~
conv2d_48/TanhTanhconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_48/Tanh�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_22/ReadVariableOp�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_22/ReadVariableOp_1�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_48/Tanh:y:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_22/FusedBatchNormV3�
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_22/AssignNewValue�
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_22/AssignNewValue_1�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_49/BiasAdd~
conv2d_49/TanhTanhconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_49/Tanh�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_23/ReadVariableOp�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_23/ReadVariableOp_1�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_49/Tanh:y:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_23/FusedBatchNormV3�
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_23/AssignNewValue�
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_23/AssignNewValue_1�
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_50/Conv2D/ReadVariableOp�
conv2d_50/Conv2DConv2D+batch_normalization_23/FusedBatchNormV3:y:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
conv2d_50/Conv2D�
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp�
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_50/BiasAdd~
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_50/Relu�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_50/Relu:activations:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_24/FusedBatchNormV3�
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_24/AssignNewValue�
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_24/AssignNewValue_1�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_51/Relu�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_51/Relu:activations:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_25/FusedBatchNormV3�
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue�
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1�
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2D+batch_normalization_25/FusedBatchNormV3:y:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_52/Relu�
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_26/ReadVariableOp�
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_26/ReadVariableOp_1�
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_52/Relu:activations:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_26/FusedBatchNormV3�
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue�
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2D+batch_normalization_26/FusedBatchNormV3:y:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_53/BiasAdd~
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_53/Relu�
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_27/ReadVariableOp�
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_27/ReadVariableOp_1�
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_53/Relu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_27/FusedBatchNormV3�
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue�
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1�
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_54/Conv2D/ReadVariableOp�
conv2d_54/Conv2DConv2D+batch_normalization_27/FusedBatchNormV3:y:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
conv2d_54/Conv2D�
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp�
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
conv2d_54/Relu�
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp�
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1�
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_54/Relu:activations:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_28/FusedBatchNormV3�
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue�
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1z
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axis�
concatenate_10/concatConcatV2+batch_normalization_28/FusedBatchNormV3:y:0inputs#concatenate_10/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������		)2
concatenate_10/concat�
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:)	*
dtype02!
conv2d_55/Conv2D/ReadVariableOp�
conv2d_55/Conv2DConv2Dconcatenate_10/concat:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			*
paddingSAME*
strides
2
conv2d_55/Conv2D�
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp�
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			2
conv2d_55/BiasAdd�
#softmax_map_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#softmax_map_4/Max/reduction_indices�
softmax_map_4/MaxMaxconv2d_55/BiasAdd:output:0,softmax_map_4/Max/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
softmax_map_4/Max�
softmax_map_4/subSubconv2d_55/BiasAdd:output:0softmax_map_4/Max:output:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/sub~
softmax_map_4/ExpExpsoftmax_map_4/sub:z:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/Exp�
#softmax_map_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#softmax_map_4/Sum/reduction_indices�
softmax_map_4/SumSumsoftmax_map_4/Exp:y:0,softmax_map_4/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
softmax_map_4/Sum�
softmax_map_4/truedivRealDivsoftmax_map_4/Exp:y:0softmax_map_4/Sum:output:0*
T0*/
_output_shapes
:���������			2
softmax_map_4/truediv�
IdentityIdentitysoftmax_map_4/truediv:z:0&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243675

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_27_layer_call_fn_244459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2408792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243693

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243905

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244331

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_240744

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
I__inference_concatenate_9_layer_call_and_return_conditional_losses_241115

inputs
inputs_1
inputs_2
inputs_3
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:���������		�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*
_input_shapesn
l:���������		 :���������		 :���������		 :���������		 :W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������		 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������		 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_49_layer_call_and_return_conditional_losses_243794

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244415

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�A
__inference__traced_save_245171
file_prefix/
+savev2_conv2d_44_kernel_read_readvariableop-
)savev2_conv2d_44_bias_read_readvariableop/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_44_kernel_m_read_readvariableop4
0savev2_adam_conv2d_44_bias_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_46_kernel_m_read_readvariableop4
0savev2_adam_conv2d_46_bias_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_m_read_readvariableop6
2savev2_adam_conv2d_48_kernel_m_read_readvariableop4
0savev2_adam_conv2d_48_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_m_read_readvariableop6
2savev2_adam_conv2d_49_kernel_m_read_readvariableop4
0savev2_adam_conv2d_49_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop6
2savev2_adam_conv2d_50_kernel_m_read_readvariableop4
0savev2_adam_conv2d_50_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_m_read_readvariableop6
2savev2_adam_conv2d_52_kernel_m_read_readvariableop4
0savev2_adam_conv2d_52_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_m_read_readvariableop6
2savev2_adam_conv2d_53_kernel_m_read_readvariableop4
0savev2_adam_conv2d_53_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_m_read_readvariableop6
2savev2_adam_conv2d_54_kernel_m_read_readvariableop4
0savev2_adam_conv2d_54_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_28_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_28_beta_m_read_readvariableop6
2savev2_adam_conv2d_55_kernel_m_read_readvariableop4
0savev2_adam_conv2d_55_bias_m_read_readvariableop6
2savev2_adam_conv2d_44_kernel_v_read_readvariableop4
0savev2_adam_conv2d_44_bias_v_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableop6
2savev2_adam_conv2d_46_kernel_v_read_readvariableop4
0savev2_adam_conv2d_46_bias_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_v_read_readvariableop6
2savev2_adam_conv2d_48_kernel_v_read_readvariableop4
0savev2_adam_conv2d_48_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_v_read_readvariableop6
2savev2_adam_conv2d_49_kernel_v_read_readvariableop4
0savev2_adam_conv2d_49_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop6
2savev2_adam_conv2d_50_kernel_v_read_readvariableop4
0savev2_adam_conv2d_50_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_24_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_24_beta_v_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_25_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_25_beta_v_read_readvariableop6
2savev2_adam_conv2d_52_kernel_v_read_readvariableop4
0savev2_adam_conv2d_52_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_26_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_26_beta_v_read_readvariableop6
2savev2_adam_conv2d_53_kernel_v_read_readvariableop4
0savev2_adam_conv2d_53_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_27_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_27_beta_v_read_readvariableop6
2savev2_adam_conv2d_54_kernel_v_read_readvariableop4
0savev2_adam_conv2d_54_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_28_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_28_beta_v_read_readvariableop6
2savev2_adam_conv2d_55_kernel_v_read_readvariableop4
0savev2_adam_conv2d_55_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�Q
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�P
value�PB�P�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_44_kernel_read_readvariableop)savev2_conv2d_44_bias_read_readvariableop+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_44_kernel_m_read_readvariableop0savev2_adam_conv2d_44_bias_m_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop2savev2_adam_conv2d_46_kernel_m_read_readvariableop0savev2_adam_conv2d_46_bias_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop>savev2_adam_batch_normalization_21_gamma_m_read_readvariableop=savev2_adam_batch_normalization_21_beta_m_read_readvariableop2savev2_adam_conv2d_48_kernel_m_read_readvariableop0savev2_adam_conv2d_48_bias_m_read_readvariableop>savev2_adam_batch_normalization_22_gamma_m_read_readvariableop=savev2_adam_batch_normalization_22_beta_m_read_readvariableop2savev2_adam_conv2d_49_kernel_m_read_readvariableop0savev2_adam_conv2d_49_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop2savev2_adam_conv2d_50_kernel_m_read_readvariableop0savev2_adam_conv2d_50_bias_m_read_readvariableop>savev2_adam_batch_normalization_24_gamma_m_read_readvariableop=savev2_adam_batch_normalization_24_beta_m_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop>savev2_adam_batch_normalization_25_gamma_m_read_readvariableop=savev2_adam_batch_normalization_25_beta_m_read_readvariableop2savev2_adam_conv2d_52_kernel_m_read_readvariableop0savev2_adam_conv2d_52_bias_m_read_readvariableop>savev2_adam_batch_normalization_26_gamma_m_read_readvariableop=savev2_adam_batch_normalization_26_beta_m_read_readvariableop2savev2_adam_conv2d_53_kernel_m_read_readvariableop0savev2_adam_conv2d_53_bias_m_read_readvariableop>savev2_adam_batch_normalization_27_gamma_m_read_readvariableop=savev2_adam_batch_normalization_27_beta_m_read_readvariableop2savev2_adam_conv2d_54_kernel_m_read_readvariableop0savev2_adam_conv2d_54_bias_m_read_readvariableop>savev2_adam_batch_normalization_28_gamma_m_read_readvariableop=savev2_adam_batch_normalization_28_beta_m_read_readvariableop2savev2_adam_conv2d_55_kernel_m_read_readvariableop0savev2_adam_conv2d_55_bias_m_read_readvariableop2savev2_adam_conv2d_44_kernel_v_read_readvariableop0savev2_adam_conv2d_44_bias_v_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop2savev2_adam_conv2d_46_kernel_v_read_readvariableop0savev2_adam_conv2d_46_bias_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop>savev2_adam_batch_normalization_21_gamma_v_read_readvariableop=savev2_adam_batch_normalization_21_beta_v_read_readvariableop2savev2_adam_conv2d_48_kernel_v_read_readvariableop0savev2_adam_conv2d_48_bias_v_read_readvariableop>savev2_adam_batch_normalization_22_gamma_v_read_readvariableop=savev2_adam_batch_normalization_22_beta_v_read_readvariableop2savev2_adam_conv2d_49_kernel_v_read_readvariableop0savev2_adam_conv2d_49_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop2savev2_adam_conv2d_50_kernel_v_read_readvariableop0savev2_adam_conv2d_50_bias_v_read_readvariableop>savev2_adam_batch_normalization_24_gamma_v_read_readvariableop=savev2_adam_batch_normalization_24_beta_v_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop>savev2_adam_batch_normalization_25_gamma_v_read_readvariableop=savev2_adam_batch_normalization_25_beta_v_read_readvariableop2savev2_adam_conv2d_52_kernel_v_read_readvariableop0savev2_adam_conv2d_52_bias_v_read_readvariableop>savev2_adam_batch_normalization_26_gamma_v_read_readvariableop=savev2_adam_batch_normalization_26_beta_v_read_readvariableop2savev2_adam_conv2d_53_kernel_v_read_readvariableop0savev2_adam_conv2d_53_bias_v_read_readvariableop>savev2_adam_batch_normalization_27_gamma_v_read_readvariableop=savev2_adam_batch_normalization_27_beta_v_read_readvariableop2savev2_adam_conv2d_54_kernel_v_read_readvariableop0savev2_adam_conv2d_54_bias_v_read_readvariableop>savev2_adam_batch_normalization_28_gamma_v_read_readvariableop=savev2_adam_batch_normalization_28_beta_v_read_readvariableop2savev2_adam_conv2d_55_kernel_v_read_readvariableop0savev2_adam_conv2d_55_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�

_input_shapes�

�	: :	 : :		 : :		 : :			 : :�:�:�:�:		�@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@ : : : : : :  : : : : : :  : : : : : :  : : : : : :)	:	: : : : : : : :	 : :		 : :		 : :			 : :�:�:		�@:@:@:@:@@:@:@:@:@@:@:@:@:@ : : : :  : : : :  : : : :  : : : :)	:	:	 : :		 : :		 : :			 : :�:�:		�@:@:@:@:@@:@:@:@:@@:@:@:@:@ : : : :  : : : :  : : : :  : : : :)	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:	 : 

_output_shapes
: :,(
&
_output_shapes
:		 : 

_output_shapes
: :,(
&
_output_shapes
:		 : 

_output_shapes
: :,(
&
_output_shapes
:			 : 

_output_shapes
: :!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:		�@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:  : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :,+(
&
_output_shapes
:  : ,

_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: :,1(
&
_output_shapes
:  : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :,7(
&
_output_shapes
:)	: 8

_output_shapes
:	:9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :,@(
&
_output_shapes
:	 : A

_output_shapes
: :,B(
&
_output_shapes
:		 : C

_output_shapes
: :,D(
&
_output_shapes
:		 : E

_output_shapes
: :,F(
&
_output_shapes
:			 : G

_output_shapes
: :!H

_output_shapes	
:�:!I

_output_shapes	
:�:-J)
'
_output_shapes
:		�@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:,N(
&
_output_shapes
:@@: O

_output_shapes
:@: P

_output_shapes
:@: Q

_output_shapes
:@:,R(
&
_output_shapes
:@@: S

_output_shapes
:@: T

_output_shapes
:@: U

_output_shapes
:@:,V(
&
_output_shapes
:@ : W

_output_shapes
: : X

_output_shapes
: : Y

_output_shapes
: :,Z(
&
_output_shapes
:  : [

_output_shapes
: : \

_output_shapes
: : ]

_output_shapes
: :,^(
&
_output_shapes
:  : _

_output_shapes
: : `

_output_shapes
: : a

_output_shapes
: :,b(
&
_output_shapes
:  : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: :,f(
&
_output_shapes
:)	: g

_output_shapes
:	:,h(
&
_output_shapes
:	 : i

_output_shapes
: :,j(
&
_output_shapes
:		 : k

_output_shapes
: :,l(
&
_output_shapes
:		 : m

_output_shapes
: :,n(
&
_output_shapes
:			 : o

_output_shapes
: :!p

_output_shapes	
:�:!q

_output_shapes	
:�:-r)
'
_output_shapes
:		�@: s

_output_shapes
:@: t

_output_shapes
:@: u

_output_shapes
:@:,v(
&
_output_shapes
:@@: w

_output_shapes
:@: x

_output_shapes
:@: y

_output_shapes
:@:,z(
&
_output_shapes
:@@: {

_output_shapes
:@: |

_output_shapes
:@: }

_output_shapes
:@:,~(
&
_output_shapes
:@ : 

_output_shapes
: :!�

_output_shapes
: :!�

_output_shapes
: :-�(
&
_output_shapes
:  :!�

_output_shapes
: :!�

_output_shapes
: :!�

_output_shapes
: :-�(
&
_output_shapes
:  :!�

_output_shapes
: :!�

_output_shapes
: :!�

_output_shapes
: :-�(
&
_output_shapes
:  :!�

_output_shapes
: :!�

_output_shapes
: :!�

_output_shapes
: :-�(
&
_output_shapes
:)	:!�

_output_shapes
:	:�

_output_shapes
: 
�
�
7__inference_batch_normalization_22_layer_call_fn_243719

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_2412632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_27_layer_call_fn_244446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2408482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
C__inference_model_4_layer_call_and_return_conditional_losses_242100
input_5
conv2d_44_241964
conv2d_44_241966
conv2d_45_241969
conv2d_45_241971
conv2d_46_241974
conv2d_46_241976
conv2d_47_241979
conv2d_47_241981!
batch_normalization_21_241985!
batch_normalization_21_241987!
batch_normalization_21_241989!
batch_normalization_21_241991
conv2d_48_241994
conv2d_48_241996!
batch_normalization_22_241999!
batch_normalization_22_242001!
batch_normalization_22_242003!
batch_normalization_22_242005
conv2d_49_242008
conv2d_49_242010!
batch_normalization_23_242013!
batch_normalization_23_242015!
batch_normalization_23_242017!
batch_normalization_23_242019
conv2d_50_242022
conv2d_50_242024!
batch_normalization_24_242027!
batch_normalization_24_242029!
batch_normalization_24_242031!
batch_normalization_24_242033
conv2d_51_242036
conv2d_51_242038!
batch_normalization_25_242041!
batch_normalization_25_242043!
batch_normalization_25_242045!
batch_normalization_25_242047
conv2d_52_242050
conv2d_52_242052!
batch_normalization_26_242055!
batch_normalization_26_242057!
batch_normalization_26_242059!
batch_normalization_26_242061
conv2d_53_242064
conv2d_53_242066!
batch_normalization_27_242069!
batch_normalization_27_242071!
batch_normalization_27_242073!
batch_normalization_27_242075
conv2d_54_242078
conv2d_54_242080!
batch_normalization_28_242083!
batch_normalization_28_242085!
batch_normalization_28_242087!
batch_normalization_28_242089
conv2d_55_242093
conv2d_55_242095
identity��.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall�.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall�!conv2d_44/StatefulPartitionedCall�!conv2d_45/StatefulPartitionedCall�!conv2d_46/StatefulPartitionedCall�!conv2d_47/StatefulPartitionedCall�!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall�
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_44_241964conv2d_44_241966*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_2410092#
!conv2d_44/StatefulPartitionedCall�
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_45_241969conv2d_45_241971*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_2410362#
!conv2d_45/StatefulPartitionedCall�
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_46_241974conv2d_46_241976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_2410632#
!conv2d_46/StatefulPartitionedCall�
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_47_241979conv2d_47_241981*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_2410902#
!conv2d_47/StatefulPartitionedCall�
concatenate_9/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*conv2d_45/StatefulPartitionedCall:output:0*conv2d_46/StatefulPartitionedCall:output:0*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_9_layer_call_and_return_conditional_losses_2411152
concatenate_9/PartitionedCall�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0batch_normalization_21_241985batch_normalization_21_241987batch_normalization_21_241989batch_normalization_21_241991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_24116320
.batch_normalization_21/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0conv2d_48_241994conv2d_48_241996*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_48_layer_call_and_return_conditional_losses_2412102#
!conv2d_48/StatefulPartitionedCall�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_22_241999batch_normalization_22_242001batch_normalization_22_242003batch_normalization_22_242005*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_24126320
.batch_normalization_22/StatefulPartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_49_242008conv2d_49_242010*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_2413102#
!conv2d_49/StatefulPartitionedCall�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_23_242013batch_normalization_23_242015batch_normalization_23_242017batch_normalization_23_242019*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_24136320
.batch_normalization_23/StatefulPartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_50_242022conv2d_50_242024*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_2414102#
!conv2d_50/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_24_242027batch_normalization_24_242029batch_normalization_24_242031batch_normalization_24_242033*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_24146320
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_51_242036conv2d_51_242038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2415102#
!conv2d_51/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_25_242041batch_normalization_25_242043batch_normalization_25_242045batch_normalization_25_242047*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_24156320
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_52_242050conv2d_52_242052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2416102#
!conv2d_52/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_26_242055batch_normalization_26_242057batch_normalization_26_242059batch_normalization_26_242061*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_24166320
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_53_242064conv2d_53_242066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2417102#
!conv2d_53/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_27_242069batch_normalization_27_242071batch_normalization_27_242073batch_normalization_27_242075*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_24176320
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_54_242078conv2d_54_242080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_2418102#
!conv2d_54/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_28_242083batch_normalization_28_242085batch_normalization_28_242087batch_normalization_28_242089*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_24186320
.batch_normalization_28/StatefulPartitionedCall�
concatenate_10/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0input_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_10_layer_call_and_return_conditional_losses_2419062 
concatenate_10/PartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_55_242093conv2d_55_242095*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_2419252#
!conv2d_55/StatefulPartitionedCall�
softmax_map_4/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_2419522
softmax_map_4/PartitionedCall�
IdentityIdentity&softmax_map_4/PartitionedCall:output:0/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244285

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_242740
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_2401622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�
�
(__inference_model_4_layer_call_fn_243293

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*J
_read_only_resource_inputs,
*(	
 !"%&'(+,-.123478*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2422422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_241545

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_26_layer_call_fn_244311

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_2407752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_53_layer_call_and_return_conditional_losses_244386

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243739

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_240640

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243591

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243545

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
I__inference_concatenate_9_layer_call_and_return_conditional_losses_243499
inputs_0
inputs_1
inputs_2
inputs_3
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:���������		�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*
_input_shapesn
l:���������		 :���������		 :���������		 :���������		 :Y U
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/3
�

*__inference_conv2d_47_layer_call_fn_243490

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_2410902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
��
�R
"__inference__traced_restore_245610
file_prefix%
!assignvariableop_conv2d_44_kernel%
!assignvariableop_1_conv2d_44_bias'
#assignvariableop_2_conv2d_45_kernel%
!assignvariableop_3_conv2d_45_bias'
#assignvariableop_4_conv2d_46_kernel%
!assignvariableop_5_conv2d_46_bias'
#assignvariableop_6_conv2d_47_kernel%
!assignvariableop_7_conv2d_47_bias3
/assignvariableop_8_batch_normalization_21_gamma2
.assignvariableop_9_batch_normalization_21_beta:
6assignvariableop_10_batch_normalization_21_moving_mean>
:assignvariableop_11_batch_normalization_21_moving_variance(
$assignvariableop_12_conv2d_48_kernel&
"assignvariableop_13_conv2d_48_bias4
0assignvariableop_14_batch_normalization_22_gamma3
/assignvariableop_15_batch_normalization_22_beta:
6assignvariableop_16_batch_normalization_22_moving_mean>
:assignvariableop_17_batch_normalization_22_moving_variance(
$assignvariableop_18_conv2d_49_kernel&
"assignvariableop_19_conv2d_49_bias4
0assignvariableop_20_batch_normalization_23_gamma3
/assignvariableop_21_batch_normalization_23_beta:
6assignvariableop_22_batch_normalization_23_moving_mean>
:assignvariableop_23_batch_normalization_23_moving_variance(
$assignvariableop_24_conv2d_50_kernel&
"assignvariableop_25_conv2d_50_bias4
0assignvariableop_26_batch_normalization_24_gamma3
/assignvariableop_27_batch_normalization_24_beta:
6assignvariableop_28_batch_normalization_24_moving_mean>
:assignvariableop_29_batch_normalization_24_moving_variance(
$assignvariableop_30_conv2d_51_kernel&
"assignvariableop_31_conv2d_51_bias4
0assignvariableop_32_batch_normalization_25_gamma3
/assignvariableop_33_batch_normalization_25_beta:
6assignvariableop_34_batch_normalization_25_moving_mean>
:assignvariableop_35_batch_normalization_25_moving_variance(
$assignvariableop_36_conv2d_52_kernel&
"assignvariableop_37_conv2d_52_bias4
0assignvariableop_38_batch_normalization_26_gamma3
/assignvariableop_39_batch_normalization_26_beta:
6assignvariableop_40_batch_normalization_26_moving_mean>
:assignvariableop_41_batch_normalization_26_moving_variance(
$assignvariableop_42_conv2d_53_kernel&
"assignvariableop_43_conv2d_53_bias4
0assignvariableop_44_batch_normalization_27_gamma3
/assignvariableop_45_batch_normalization_27_beta:
6assignvariableop_46_batch_normalization_27_moving_mean>
:assignvariableop_47_batch_normalization_27_moving_variance(
$assignvariableop_48_conv2d_54_kernel&
"assignvariableop_49_conv2d_54_bias4
0assignvariableop_50_batch_normalization_28_gamma3
/assignvariableop_51_batch_normalization_28_beta:
6assignvariableop_52_batch_normalization_28_moving_mean>
:assignvariableop_53_batch_normalization_28_moving_variance(
$assignvariableop_54_conv2d_55_kernel&
"assignvariableop_55_conv2d_55_bias!
assignvariableop_56_adam_iter#
assignvariableop_57_adam_beta_1#
assignvariableop_58_adam_beta_2"
assignvariableop_59_adam_decay*
&assignvariableop_60_adam_learning_rate
assignvariableop_61_total
assignvariableop_62_count/
+assignvariableop_63_adam_conv2d_44_kernel_m-
)assignvariableop_64_adam_conv2d_44_bias_m/
+assignvariableop_65_adam_conv2d_45_kernel_m-
)assignvariableop_66_adam_conv2d_45_bias_m/
+assignvariableop_67_adam_conv2d_46_kernel_m-
)assignvariableop_68_adam_conv2d_46_bias_m/
+assignvariableop_69_adam_conv2d_47_kernel_m-
)assignvariableop_70_adam_conv2d_47_bias_m;
7assignvariableop_71_adam_batch_normalization_21_gamma_m:
6assignvariableop_72_adam_batch_normalization_21_beta_m/
+assignvariableop_73_adam_conv2d_48_kernel_m-
)assignvariableop_74_adam_conv2d_48_bias_m;
7assignvariableop_75_adam_batch_normalization_22_gamma_m:
6assignvariableop_76_adam_batch_normalization_22_beta_m/
+assignvariableop_77_adam_conv2d_49_kernel_m-
)assignvariableop_78_adam_conv2d_49_bias_m;
7assignvariableop_79_adam_batch_normalization_23_gamma_m:
6assignvariableop_80_adam_batch_normalization_23_beta_m/
+assignvariableop_81_adam_conv2d_50_kernel_m-
)assignvariableop_82_adam_conv2d_50_bias_m;
7assignvariableop_83_adam_batch_normalization_24_gamma_m:
6assignvariableop_84_adam_batch_normalization_24_beta_m/
+assignvariableop_85_adam_conv2d_51_kernel_m-
)assignvariableop_86_adam_conv2d_51_bias_m;
7assignvariableop_87_adam_batch_normalization_25_gamma_m:
6assignvariableop_88_adam_batch_normalization_25_beta_m/
+assignvariableop_89_adam_conv2d_52_kernel_m-
)assignvariableop_90_adam_conv2d_52_bias_m;
7assignvariableop_91_adam_batch_normalization_26_gamma_m:
6assignvariableop_92_adam_batch_normalization_26_beta_m/
+assignvariableop_93_adam_conv2d_53_kernel_m-
)assignvariableop_94_adam_conv2d_53_bias_m;
7assignvariableop_95_adam_batch_normalization_27_gamma_m:
6assignvariableop_96_adam_batch_normalization_27_beta_m/
+assignvariableop_97_adam_conv2d_54_kernel_m-
)assignvariableop_98_adam_conv2d_54_bias_m;
7assignvariableop_99_adam_batch_normalization_28_gamma_m;
7assignvariableop_100_adam_batch_normalization_28_beta_m0
,assignvariableop_101_adam_conv2d_55_kernel_m.
*assignvariableop_102_adam_conv2d_55_bias_m0
,assignvariableop_103_adam_conv2d_44_kernel_v.
*assignvariableop_104_adam_conv2d_44_bias_v0
,assignvariableop_105_adam_conv2d_45_kernel_v.
*assignvariableop_106_adam_conv2d_45_bias_v0
,assignvariableop_107_adam_conv2d_46_kernel_v.
*assignvariableop_108_adam_conv2d_46_bias_v0
,assignvariableop_109_adam_conv2d_47_kernel_v.
*assignvariableop_110_adam_conv2d_47_bias_v<
8assignvariableop_111_adam_batch_normalization_21_gamma_v;
7assignvariableop_112_adam_batch_normalization_21_beta_v0
,assignvariableop_113_adam_conv2d_48_kernel_v.
*assignvariableop_114_adam_conv2d_48_bias_v<
8assignvariableop_115_adam_batch_normalization_22_gamma_v;
7assignvariableop_116_adam_batch_normalization_22_beta_v0
,assignvariableop_117_adam_conv2d_49_kernel_v.
*assignvariableop_118_adam_conv2d_49_bias_v<
8assignvariableop_119_adam_batch_normalization_23_gamma_v;
7assignvariableop_120_adam_batch_normalization_23_beta_v0
,assignvariableop_121_adam_conv2d_50_kernel_v.
*assignvariableop_122_adam_conv2d_50_bias_v<
8assignvariableop_123_adam_batch_normalization_24_gamma_v;
7assignvariableop_124_adam_batch_normalization_24_beta_v0
,assignvariableop_125_adam_conv2d_51_kernel_v.
*assignvariableop_126_adam_conv2d_51_bias_v<
8assignvariableop_127_adam_batch_normalization_25_gamma_v;
7assignvariableop_128_adam_batch_normalization_25_beta_v0
,assignvariableop_129_adam_conv2d_52_kernel_v.
*assignvariableop_130_adam_conv2d_52_bias_v<
8assignvariableop_131_adam_batch_normalization_26_gamma_v;
7assignvariableop_132_adam_batch_normalization_26_beta_v0
,assignvariableop_133_adam_conv2d_53_kernel_v.
*assignvariableop_134_adam_conv2d_53_bias_v<
8assignvariableop_135_adam_batch_normalization_27_gamma_v;
7assignvariableop_136_adam_batch_normalization_27_beta_v0
,assignvariableop_137_adam_conv2d_54_kernel_v.
*assignvariableop_138_adam_conv2d_54_bias_v<
8assignvariableop_139_adam_batch_normalization_28_gamma_v;
7assignvariableop_140_adam_batch_normalization_28_beta_v0
,assignvariableop_141_adam_conv2d_55_kernel_v.
*assignvariableop_142_adam_conv2d_55_bias_v
identity_144��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�Q
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�P
value�PB�P�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_44_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_44_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_45_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_45_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_46_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_46_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_47_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_47_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_21_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_21_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_21_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_21_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_48_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_48_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_22_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_22_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_22_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_22_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_49_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_49_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_23_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_23_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_23_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_23_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_50_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_50_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_24_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_24_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_24_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_24_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_51_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_51_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_25_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_25_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_25_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_25_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_52_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_52_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_26_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_26_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_26_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_26_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_53_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv2d_53_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_27_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_27_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_27_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_27_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_54_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv2d_54_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp0assignvariableop_50_batch_normalization_28_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp/assignvariableop_51_batch_normalization_28_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp6assignvariableop_52_batch_normalization_28_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp:assignvariableop_53_batch_normalization_28_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp$assignvariableop_54_conv2d_55_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp"assignvariableop_55_conv2d_55_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_iterIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_beta_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_beta_2Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_decayIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_learning_rateIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOpassignvariableop_61_totalIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOpassignvariableop_62_countIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_44_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_44_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_45_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_45_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_46_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_46_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_47_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_47_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_21_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_21_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_48_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_48_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_22_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_22_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_49_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_49_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_23_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_23_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_50_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_50_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_24_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_24_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_51_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_51_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_25_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_25_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv2d_52_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv2d_52_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91�
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_26_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92�
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_26_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93�
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv2d_53_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94�
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv2d_53_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95�
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_27_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96�
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_27_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97�
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv2d_54_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98�
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv2d_54_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99�
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_28_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100�
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_28_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101�
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_55_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102�
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_55_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103�
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv2d_44_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104�
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv2d_44_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105�
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_45_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106�
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_45_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107�
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv2d_46_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108�
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv2d_46_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109�
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_47_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110�
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_47_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111�
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_21_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112�
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_21_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113�
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_48_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114�
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_48_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115�
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_22_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116�
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_22_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117�
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_49_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118�
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_49_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119�
AssignVariableOp_119AssignVariableOp8assignvariableop_119_adam_batch_normalization_23_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120�
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adam_batch_normalization_23_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121�
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv2d_50_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122�
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv2d_50_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123�
AssignVariableOp_123AssignVariableOp8assignvariableop_123_adam_batch_normalization_24_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124�
AssignVariableOp_124AssignVariableOp7assignvariableop_124_adam_batch_normalization_24_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125�
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv2d_51_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126�
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv2d_51_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127�
AssignVariableOp_127AssignVariableOp8assignvariableop_127_adam_batch_normalization_25_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128�
AssignVariableOp_128AssignVariableOp7assignvariableop_128_adam_batch_normalization_25_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129�
AssignVariableOp_129AssignVariableOp,assignvariableop_129_adam_conv2d_52_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130�
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adam_conv2d_52_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131�
AssignVariableOp_131AssignVariableOp8assignvariableop_131_adam_batch_normalization_26_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132�
AssignVariableOp_132AssignVariableOp7assignvariableop_132_adam_batch_normalization_26_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133�
AssignVariableOp_133AssignVariableOp,assignvariableop_133_adam_conv2d_53_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134�
AssignVariableOp_134AssignVariableOp*assignvariableop_134_adam_conv2d_53_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135�
AssignVariableOp_135AssignVariableOp8assignvariableop_135_adam_batch_normalization_27_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136�
AssignVariableOp_136AssignVariableOp7assignvariableop_136_adam_batch_normalization_27_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137�
AssignVariableOp_137AssignVariableOp,assignvariableop_137_adam_conv2d_54_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138�
AssignVariableOp_138AssignVariableOp*assignvariableop_138_adam_conv2d_54_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139�
AssignVariableOp_139AssignVariableOp8assignvariableop_139_adam_batch_normalization_28_gamma_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140�
AssignVariableOp_140AssignVariableOp7assignvariableop_140_adam_batch_normalization_28_beta_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141�
AssignVariableOp_141AssignVariableOp,assignvariableop_141_adam_conv2d_55_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142�
AssignVariableOp_142AssignVariableOp*assignvariableop_142_adam_conv2d_55_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_143Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_143�
Identity_144IdentityIdentity_143:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_144"%
identity_144Identity_144:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244627

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

*__inference_conv2d_53_layer_call_fn_244395

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2417102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
΅
�
C__inference_model_4_layer_call_and_return_conditional_losses_242242

inputs
conv2d_44_242106
conv2d_44_242108
conv2d_45_242111
conv2d_45_242113
conv2d_46_242116
conv2d_46_242118
conv2d_47_242121
conv2d_47_242123!
batch_normalization_21_242127!
batch_normalization_21_242129!
batch_normalization_21_242131!
batch_normalization_21_242133
conv2d_48_242136
conv2d_48_242138!
batch_normalization_22_242141!
batch_normalization_22_242143!
batch_normalization_22_242145!
batch_normalization_22_242147
conv2d_49_242150
conv2d_49_242152!
batch_normalization_23_242155!
batch_normalization_23_242157!
batch_normalization_23_242159!
batch_normalization_23_242161
conv2d_50_242164
conv2d_50_242166!
batch_normalization_24_242169!
batch_normalization_24_242171!
batch_normalization_24_242173!
batch_normalization_24_242175
conv2d_51_242178
conv2d_51_242180!
batch_normalization_25_242183!
batch_normalization_25_242185!
batch_normalization_25_242187!
batch_normalization_25_242189
conv2d_52_242192
conv2d_52_242194!
batch_normalization_26_242197!
batch_normalization_26_242199!
batch_normalization_26_242201!
batch_normalization_26_242203
conv2d_53_242206
conv2d_53_242208!
batch_normalization_27_242211!
batch_normalization_27_242213!
batch_normalization_27_242215!
batch_normalization_27_242217
conv2d_54_242220
conv2d_54_242222!
batch_normalization_28_242225!
batch_normalization_28_242227!
batch_normalization_28_242229!
batch_normalization_28_242231
conv2d_55_242235
conv2d_55_242237
identity��.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall�.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall�!conv2d_44/StatefulPartitionedCall�!conv2d_45/StatefulPartitionedCall�!conv2d_46/StatefulPartitionedCall�!conv2d_47/StatefulPartitionedCall�!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall�
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_44_242106conv2d_44_242108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_2410092#
!conv2d_44/StatefulPartitionedCall�
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45_242111conv2d_45_242113*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_2410362#
!conv2d_45/StatefulPartitionedCall�
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_46_242116conv2d_46_242118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_2410632#
!conv2d_46/StatefulPartitionedCall�
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_47_242121conv2d_47_242123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_2410902#
!conv2d_47/StatefulPartitionedCall�
concatenate_9/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*conv2d_45/StatefulPartitionedCall:output:0*conv2d_46/StatefulPartitionedCall:output:0*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_9_layer_call_and_return_conditional_losses_2411152
concatenate_9/PartitionedCall�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0batch_normalization_21_242127batch_normalization_21_242129batch_normalization_21_242131batch_normalization_21_242133*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_24114520
.batch_normalization_21/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0conv2d_48_242136conv2d_48_242138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_48_layer_call_and_return_conditional_losses_2412102#
!conv2d_48/StatefulPartitionedCall�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_22_242141batch_normalization_22_242143batch_normalization_22_242145batch_normalization_22_242147*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_24124520
.batch_normalization_22/StatefulPartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_49_242150conv2d_49_242152*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_2413102#
!conv2d_49/StatefulPartitionedCall�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_23_242155batch_normalization_23_242157batch_normalization_23_242159batch_normalization_23_242161*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_24134520
.batch_normalization_23/StatefulPartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_50_242164conv2d_50_242166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_2414102#
!conv2d_50/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_24_242169batch_normalization_24_242171batch_normalization_24_242173batch_normalization_24_242175*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_24144520
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_51_242178conv2d_51_242180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2415102#
!conv2d_51/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_25_242183batch_normalization_25_242185batch_normalization_25_242187batch_normalization_25_242189*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_24154520
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_52_242192conv2d_52_242194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2416102#
!conv2d_52/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_26_242197batch_normalization_26_242199batch_normalization_26_242201batch_normalization_26_242203*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_24164520
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_53_242206conv2d_53_242208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2417102#
!conv2d_53/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_27_242211batch_normalization_27_242213batch_normalization_27_242215batch_normalization_27_242217*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_24174520
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_54_242220conv2d_54_242222*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_2418102#
!conv2d_54/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_28_242225batch_normalization_28_242227batch_normalization_28_242229batch_normalization_28_242231*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_24184520
.batch_normalization_28/StatefulPartitionedCall�
concatenate_10/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_10_layer_call_and_return_conditional_losses_2419062 
concatenate_10/PartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_55_242235conv2d_55_242237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_2419252#
!conv2d_55/StatefulPartitionedCall�
softmax_map_4/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_2419522
softmax_map_4/PartitionedCall�
IdentityIdentity&softmax_map_4/PartitionedCall:output:0/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�

*__inference_conv2d_51_layer_call_fn_244099

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2415102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_28_layer_call_fn_244594

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2418452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�	
`
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_241952
x
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max/reduction_indices
MaxMaxxMax/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
Max\
subSubxMax:output:0*
T0*/
_output_shapes
:���������			2
subT
ExpExpsub:z:0*
T0*/
_output_shapes
:���������			2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
Sumn
truedivRealDivExp:y:0Sum:output:0*
T0*/
_output_shapes
:���������			2	
truedivg
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������			:R N
/
_output_shapes
:���������			

_user_specified_namex
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_241663

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

*__inference_conv2d_48_layer_call_fn_243655

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_48_layer_call_and_return_conditional_losses_2412102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������		�::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243887

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_24_layer_call_fn_244066

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_2414452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
��
�2
!__inference__wrapped_model_240162
input_54
0model_4_conv2d_44_conv2d_readvariableop_resource5
1model_4_conv2d_44_biasadd_readvariableop_resource4
0model_4_conv2d_45_conv2d_readvariableop_resource5
1model_4_conv2d_45_biasadd_readvariableop_resource4
0model_4_conv2d_46_conv2d_readvariableop_resource5
1model_4_conv2d_46_biasadd_readvariableop_resource4
0model_4_conv2d_47_conv2d_readvariableop_resource5
1model_4_conv2d_47_biasadd_readvariableop_resource:
6model_4_batch_normalization_21_readvariableop_resource<
8model_4_batch_normalization_21_readvariableop_1_resourceK
Gmodel_4_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_48_conv2d_readvariableop_resource5
1model_4_conv2d_48_biasadd_readvariableop_resource:
6model_4_batch_normalization_22_readvariableop_resource<
8model_4_batch_normalization_22_readvariableop_1_resourceK
Gmodel_4_batch_normalization_22_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_49_conv2d_readvariableop_resource5
1model_4_conv2d_49_biasadd_readvariableop_resource:
6model_4_batch_normalization_23_readvariableop_resource<
8model_4_batch_normalization_23_readvariableop_1_resourceK
Gmodel_4_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_50_conv2d_readvariableop_resource5
1model_4_conv2d_50_biasadd_readvariableop_resource:
6model_4_batch_normalization_24_readvariableop_resource<
8model_4_batch_normalization_24_readvariableop_1_resourceK
Gmodel_4_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_51_conv2d_readvariableop_resource5
1model_4_conv2d_51_biasadd_readvariableop_resource:
6model_4_batch_normalization_25_readvariableop_resource<
8model_4_batch_normalization_25_readvariableop_1_resourceK
Gmodel_4_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_52_conv2d_readvariableop_resource5
1model_4_conv2d_52_biasadd_readvariableop_resource:
6model_4_batch_normalization_26_readvariableop_resource<
8model_4_batch_normalization_26_readvariableop_1_resourceK
Gmodel_4_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_53_conv2d_readvariableop_resource5
1model_4_conv2d_53_biasadd_readvariableop_resource:
6model_4_batch_normalization_27_readvariableop_resource<
8model_4_batch_normalization_27_readvariableop_1_resourceK
Gmodel_4_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_54_conv2d_readvariableop_resource5
1model_4_conv2d_54_biasadd_readvariableop_resource:
6model_4_batch_normalization_28_readvariableop_resource<
8model_4_batch_normalization_28_readvariableop_1_resourceK
Gmodel_4_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceM
Imodel_4_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource4
0model_4_conv2d_55_conv2d_readvariableop_resource5
1model_4_conv2d_55_biasadd_readvariableop_resource
identity��>model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_21/ReadVariableOp�/model_4/batch_normalization_21/ReadVariableOp_1�>model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_22/ReadVariableOp�/model_4/batch_normalization_22/ReadVariableOp_1�>model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_23/ReadVariableOp�/model_4/batch_normalization_23/ReadVariableOp_1�>model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_24/ReadVariableOp�/model_4/batch_normalization_24/ReadVariableOp_1�>model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_25/ReadVariableOp�/model_4/batch_normalization_25/ReadVariableOp_1�>model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_26/ReadVariableOp�/model_4/batch_normalization_26/ReadVariableOp_1�>model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_27/ReadVariableOp�/model_4/batch_normalization_27/ReadVariableOp_1�>model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp�@model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�-model_4/batch_normalization_28/ReadVariableOp�/model_4/batch_normalization_28/ReadVariableOp_1�(model_4/conv2d_44/BiasAdd/ReadVariableOp�'model_4/conv2d_44/Conv2D/ReadVariableOp�(model_4/conv2d_45/BiasAdd/ReadVariableOp�'model_4/conv2d_45/Conv2D/ReadVariableOp�(model_4/conv2d_46/BiasAdd/ReadVariableOp�'model_4/conv2d_46/Conv2D/ReadVariableOp�(model_4/conv2d_47/BiasAdd/ReadVariableOp�'model_4/conv2d_47/Conv2D/ReadVariableOp�(model_4/conv2d_48/BiasAdd/ReadVariableOp�'model_4/conv2d_48/Conv2D/ReadVariableOp�(model_4/conv2d_49/BiasAdd/ReadVariableOp�'model_4/conv2d_49/Conv2D/ReadVariableOp�(model_4/conv2d_50/BiasAdd/ReadVariableOp�'model_4/conv2d_50/Conv2D/ReadVariableOp�(model_4/conv2d_51/BiasAdd/ReadVariableOp�'model_4/conv2d_51/Conv2D/ReadVariableOp�(model_4/conv2d_52/BiasAdd/ReadVariableOp�'model_4/conv2d_52/Conv2D/ReadVariableOp�(model_4/conv2d_53/BiasAdd/ReadVariableOp�'model_4/conv2d_53/Conv2D/ReadVariableOp�(model_4/conv2d_54/BiasAdd/ReadVariableOp�'model_4/conv2d_54/Conv2D/ReadVariableOp�(model_4/conv2d_55/BiasAdd/ReadVariableOp�'model_4/conv2d_55/Conv2D/ReadVariableOp�
'model_4/conv2d_44/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype02)
'model_4/conv2d_44/Conv2D/ReadVariableOp�
model_4/conv2d_44/Conv2DConv2Dinput_5/model_4/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_44/Conv2D�
(model_4/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_44_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_44/BiasAdd/ReadVariableOp�
model_4/conv2d_44/BiasAddBiasAdd!model_4/conv2d_44/Conv2D:output:00model_4/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_44/BiasAdd�
model_4/conv2d_44/TanhTanh"model_4/conv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_44/Tanh�
'model_4/conv2d_45/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02)
'model_4/conv2d_45/Conv2D/ReadVariableOp�
model_4/conv2d_45/Conv2DConv2Dinput_5/model_4/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_45/Conv2D�
(model_4/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_45/BiasAdd/ReadVariableOp�
model_4/conv2d_45/BiasAddBiasAdd!model_4/conv2d_45/Conv2D:output:00model_4/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_45/BiasAdd�
model_4/conv2d_45/TanhTanh"model_4/conv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_45/Tanh�
'model_4/conv2d_46/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02)
'model_4/conv2d_46/Conv2D/ReadVariableOp�
model_4/conv2d_46/Conv2DConv2Dinput_5/model_4/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_46/Conv2D�
(model_4/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_46/BiasAdd/ReadVariableOp�
model_4/conv2d_46/BiasAddBiasAdd!model_4/conv2d_46/Conv2D:output:00model_4/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_46/BiasAdd�
model_4/conv2d_46/TanhTanh"model_4/conv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_46/Tanh�
'model_4/conv2d_47/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:			 *
dtype02)
'model_4/conv2d_47/Conv2D/ReadVariableOp�
model_4/conv2d_47/Conv2DConv2Dinput_5/model_4/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_47/Conv2D�
(model_4/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_47/BiasAdd/ReadVariableOp�
model_4/conv2d_47/BiasAddBiasAdd!model_4/conv2d_47/Conv2D:output:00model_4/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_47/BiasAdd�
model_4/conv2d_47/TanhTanh"model_4/conv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_47/Tanh�
!model_4/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_4/concatenate_9/concat/axis�
model_4/concatenate_9/concatConcatV2model_4/conv2d_44/Tanh:y:0model_4/conv2d_45/Tanh:y:0model_4/conv2d_46/Tanh:y:0model_4/conv2d_47/Tanh:y:0*model_4/concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������		�2
model_4/concatenate_9/concat�
-model_4/batch_normalization_21/ReadVariableOpReadVariableOp6model_4_batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-model_4/batch_normalization_21/ReadVariableOp�
/model_4/batch_normalization_21/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype021
/model_4/batch_normalization_21/ReadVariableOp_1�
>model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02B
@model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3%model_4/concatenate_9/concat:output:05model_4/batch_normalization_21/ReadVariableOp:value:07model_4/batch_normalization_21/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_21/FusedBatchNormV3�
'model_4/conv2d_48/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_48_conv2d_readvariableop_resource*'
_output_shapes
:		�@*
dtype02)
'model_4/conv2d_48/Conv2D/ReadVariableOp�
model_4/conv2d_48/Conv2DConv2D3model_4/batch_normalization_21/FusedBatchNormV3:y:0/model_4/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
model_4/conv2d_48/Conv2D�
(model_4/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv2d_48/BiasAdd/ReadVariableOp�
model_4/conv2d_48/BiasAddBiasAdd!model_4/conv2d_48/Conv2D:output:00model_4/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_48/BiasAdd�
model_4/conv2d_48/TanhTanh"model_4/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_48/Tanh�
-model_4/batch_normalization_22/ReadVariableOpReadVariableOp6model_4_batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_4/batch_normalization_22/ReadVariableOp�
/model_4/batch_normalization_22/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_4/batch_normalization_22/ReadVariableOp_1�
>model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3model_4/conv2d_48/Tanh:y:05model_4/batch_normalization_22/ReadVariableOp:value:07model_4/batch_normalization_22/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_22/FusedBatchNormV3�
'model_4/conv2d_49/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_4/conv2d_49/Conv2D/ReadVariableOp�
model_4/conv2d_49/Conv2DConv2D3model_4/batch_normalization_22/FusedBatchNormV3:y:0/model_4/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
model_4/conv2d_49/Conv2D�
(model_4/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv2d_49/BiasAdd/ReadVariableOp�
model_4/conv2d_49/BiasAddBiasAdd!model_4/conv2d_49/Conv2D:output:00model_4/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_49/BiasAdd�
model_4/conv2d_49/TanhTanh"model_4/conv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_49/Tanh�
-model_4/batch_normalization_23/ReadVariableOpReadVariableOp6model_4_batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_4/batch_normalization_23/ReadVariableOp�
/model_4/batch_normalization_23/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_4/batch_normalization_23/ReadVariableOp_1�
>model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3model_4/conv2d_49/Tanh:y:05model_4/batch_normalization_23/ReadVariableOp:value:07model_4/batch_normalization_23/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_23/FusedBatchNormV3�
'model_4/conv2d_50/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_4/conv2d_50/Conv2D/ReadVariableOp�
model_4/conv2d_50/Conv2DConv2D3model_4/batch_normalization_23/FusedBatchNormV3:y:0/model_4/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
model_4/conv2d_50/Conv2D�
(model_4/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_4/conv2d_50/BiasAdd/ReadVariableOp�
model_4/conv2d_50/BiasAddBiasAdd!model_4/conv2d_50/Conv2D:output:00model_4/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_50/BiasAdd�
model_4/conv2d_50/ReluRelu"model_4/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
model_4/conv2d_50/Relu�
-model_4/batch_normalization_24/ReadVariableOpReadVariableOp6model_4_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_4/batch_normalization_24/ReadVariableOp�
/model_4/batch_normalization_24/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_4/batch_normalization_24/ReadVariableOp_1�
>model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3$model_4/conv2d_50/Relu:activations:05model_4/batch_normalization_24/ReadVariableOp:value:07model_4/batch_normalization_24/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_24/FusedBatchNormV3�
'model_4/conv2d_51/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'model_4/conv2d_51/Conv2D/ReadVariableOp�
model_4/conv2d_51/Conv2DConv2D3model_4/batch_normalization_24/FusedBatchNormV3:y:0/model_4/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_51/Conv2D�
(model_4/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_51/BiasAdd/ReadVariableOp�
model_4/conv2d_51/BiasAddBiasAdd!model_4/conv2d_51/Conv2D:output:00model_4/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_51/BiasAdd�
model_4/conv2d_51/ReluRelu"model_4/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_51/Relu�
-model_4/batch_normalization_25/ReadVariableOpReadVariableOp6model_4_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_4/batch_normalization_25/ReadVariableOp�
/model_4/batch_normalization_25/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_4/batch_normalization_25/ReadVariableOp_1�
>model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3$model_4/conv2d_51/Relu:activations:05model_4/batch_normalization_25/ReadVariableOp:value:07model_4/batch_normalization_25/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_25/FusedBatchNormV3�
'model_4/conv2d_52/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_4/conv2d_52/Conv2D/ReadVariableOp�
model_4/conv2d_52/Conv2DConv2D3model_4/batch_normalization_25/FusedBatchNormV3:y:0/model_4/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_52/Conv2D�
(model_4/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_52/BiasAdd/ReadVariableOp�
model_4/conv2d_52/BiasAddBiasAdd!model_4/conv2d_52/Conv2D:output:00model_4/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_52/BiasAdd�
model_4/conv2d_52/ReluRelu"model_4/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_52/Relu�
-model_4/batch_normalization_26/ReadVariableOpReadVariableOp6model_4_batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_4/batch_normalization_26/ReadVariableOp�
/model_4/batch_normalization_26/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_4/batch_normalization_26/ReadVariableOp_1�
>model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3$model_4/conv2d_52/Relu:activations:05model_4/batch_normalization_26/ReadVariableOp:value:07model_4/batch_normalization_26/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_26/FusedBatchNormV3�
'model_4/conv2d_53/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_4/conv2d_53/Conv2D/ReadVariableOp�
model_4/conv2d_53/Conv2DConv2D3model_4/batch_normalization_26/FusedBatchNormV3:y:0/model_4/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_53/Conv2D�
(model_4/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_53/BiasAdd/ReadVariableOp�
model_4/conv2d_53/BiasAddBiasAdd!model_4/conv2d_53/Conv2D:output:00model_4/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_53/BiasAdd�
model_4/conv2d_53/ReluRelu"model_4/conv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_53/Relu�
-model_4/batch_normalization_27/ReadVariableOpReadVariableOp6model_4_batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_4/batch_normalization_27/ReadVariableOp�
/model_4/batch_normalization_27/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_4/batch_normalization_27/ReadVariableOp_1�
>model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3$model_4/conv2d_53/Relu:activations:05model_4/batch_normalization_27/ReadVariableOp:value:07model_4/batch_normalization_27/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_27/FusedBatchNormV3�
'model_4/conv2d_54/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_4/conv2d_54/Conv2D/ReadVariableOp�
model_4/conv2d_54/Conv2DConv2D3model_4/batch_normalization_27/FusedBatchNormV3:y:0/model_4/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
model_4/conv2d_54/Conv2D�
(model_4/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_4/conv2d_54/BiasAdd/ReadVariableOp�
model_4/conv2d_54/BiasAddBiasAdd!model_4/conv2d_54/Conv2D:output:00model_4/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_54/BiasAdd�
model_4/conv2d_54/ReluRelu"model_4/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
model_4/conv2d_54/Relu�
-model_4/batch_normalization_28/ReadVariableOpReadVariableOp6model_4_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_4/batch_normalization_28/ReadVariableOp�
/model_4/batch_normalization_28/ReadVariableOp_1ReadVariableOp8model_4_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_4/batch_normalization_28/ReadVariableOp_1�
>model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_4_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp�
@model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_4_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1�
/model_4/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3$model_4/conv2d_54/Relu:activations:05model_4/batch_normalization_28/ReadVariableOp:value:07model_4/batch_normalization_28/ReadVariableOp_1:value:0Fmodel_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 21
/model_4/batch_normalization_28/FusedBatchNormV3�
"model_4/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_4/concatenate_10/concat/axis�
model_4/concatenate_10/concatConcatV23model_4/batch_normalization_28/FusedBatchNormV3:y:0input_5+model_4/concatenate_10/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������		)2
model_4/concatenate_10/concat�
'model_4/conv2d_55/Conv2D/ReadVariableOpReadVariableOp0model_4_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:)	*
dtype02)
'model_4/conv2d_55/Conv2D/ReadVariableOp�
model_4/conv2d_55/Conv2DConv2D&model_4/concatenate_10/concat:output:0/model_4/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			*
paddingSAME*
strides
2
model_4/conv2d_55/Conv2D�
(model_4/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp1model_4_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02*
(model_4/conv2d_55/BiasAdd/ReadVariableOp�
model_4/conv2d_55/BiasAddBiasAdd!model_4/conv2d_55/Conv2D:output:00model_4/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������			2
model_4/conv2d_55/BiasAdd�
+model_4/softmax_map_4/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+model_4/softmax_map_4/Max/reduction_indices�
model_4/softmax_map_4/MaxMax"model_4/conv2d_55/BiasAdd:output:04model_4/softmax_map_4/Max/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
model_4/softmax_map_4/Max�
model_4/softmax_map_4/subSub"model_4/conv2d_55/BiasAdd:output:0"model_4/softmax_map_4/Max:output:0*
T0*/
_output_shapes
:���������			2
model_4/softmax_map_4/sub�
model_4/softmax_map_4/ExpExpmodel_4/softmax_map_4/sub:z:0*
T0*/
_output_shapes
:���������			2
model_4/softmax_map_4/Exp�
+model_4/softmax_map_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+model_4/softmax_map_4/Sum/reduction_indices�
model_4/softmax_map_4/SumSummodel_4/softmax_map_4/Exp:y:04model_4/softmax_map_4/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
model_4/softmax_map_4/Sum�
model_4/softmax_map_4/truedivRealDivmodel_4/softmax_map_4/Exp:y:0"model_4/softmax_map_4/Sum:output:0*
T0*/
_output_shapes
:���������			2
model_4/softmax_map_4/truediv�
IdentityIdentity!model_4/softmax_map_4/truediv:z:0?^model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_21/ReadVariableOp0^model_4/batch_normalization_21/ReadVariableOp_1?^model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_22/ReadVariableOp0^model_4/batch_normalization_22/ReadVariableOp_1?^model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_23/ReadVariableOp0^model_4/batch_normalization_23/ReadVariableOp_1?^model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_24/ReadVariableOp0^model_4/batch_normalization_24/ReadVariableOp_1?^model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_25/ReadVariableOp0^model_4/batch_normalization_25/ReadVariableOp_1?^model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_26/ReadVariableOp0^model_4/batch_normalization_26/ReadVariableOp_1?^model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_27/ReadVariableOp0^model_4/batch_normalization_27/ReadVariableOp_1?^model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOpA^model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1.^model_4/batch_normalization_28/ReadVariableOp0^model_4/batch_normalization_28/ReadVariableOp_1)^model_4/conv2d_44/BiasAdd/ReadVariableOp(^model_4/conv2d_44/Conv2D/ReadVariableOp)^model_4/conv2d_45/BiasAdd/ReadVariableOp(^model_4/conv2d_45/Conv2D/ReadVariableOp)^model_4/conv2d_46/BiasAdd/ReadVariableOp(^model_4/conv2d_46/Conv2D/ReadVariableOp)^model_4/conv2d_47/BiasAdd/ReadVariableOp(^model_4/conv2d_47/Conv2D/ReadVariableOp)^model_4/conv2d_48/BiasAdd/ReadVariableOp(^model_4/conv2d_48/Conv2D/ReadVariableOp)^model_4/conv2d_49/BiasAdd/ReadVariableOp(^model_4/conv2d_49/Conv2D/ReadVariableOp)^model_4/conv2d_50/BiasAdd/ReadVariableOp(^model_4/conv2d_50/Conv2D/ReadVariableOp)^model_4/conv2d_51/BiasAdd/ReadVariableOp(^model_4/conv2d_51/Conv2D/ReadVariableOp)^model_4/conv2d_52/BiasAdd/ReadVariableOp(^model_4/conv2d_52/Conv2D/ReadVariableOp)^model_4/conv2d_53/BiasAdd/ReadVariableOp(^model_4/conv2d_53/Conv2D/ReadVariableOp)^model_4/conv2d_54/BiasAdd/ReadVariableOp(^model_4/conv2d_54/Conv2D/ReadVariableOp)^model_4/conv2d_55/BiasAdd/ReadVariableOp(^model_4/conv2d_55/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2�
>model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_21/ReadVariableOp-model_4/batch_normalization_21/ReadVariableOp2b
/model_4/batch_normalization_21/ReadVariableOp_1/model_4/batch_normalization_21/ReadVariableOp_12�
>model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_22/ReadVariableOp-model_4/batch_normalization_22/ReadVariableOp2b
/model_4/batch_normalization_22/ReadVariableOp_1/model_4/batch_normalization_22/ReadVariableOp_12�
>model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_23/ReadVariableOp-model_4/batch_normalization_23/ReadVariableOp2b
/model_4/batch_normalization_23/ReadVariableOp_1/model_4/batch_normalization_23/ReadVariableOp_12�
>model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_24/ReadVariableOp-model_4/batch_normalization_24/ReadVariableOp2b
/model_4/batch_normalization_24/ReadVariableOp_1/model_4/batch_normalization_24/ReadVariableOp_12�
>model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_25/ReadVariableOp-model_4/batch_normalization_25/ReadVariableOp2b
/model_4/batch_normalization_25/ReadVariableOp_1/model_4/batch_normalization_25/ReadVariableOp_12�
>model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_26/ReadVariableOp-model_4/batch_normalization_26/ReadVariableOp2b
/model_4/batch_normalization_26/ReadVariableOp_1/model_4/batch_normalization_26/ReadVariableOp_12�
>model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_27/ReadVariableOp-model_4/batch_normalization_27/ReadVariableOp2b
/model_4/batch_normalization_27/ReadVariableOp_1/model_4/batch_normalization_27/ReadVariableOp_12�
>model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp>model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2�
@model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1@model_4/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12^
-model_4/batch_normalization_28/ReadVariableOp-model_4/batch_normalization_28/ReadVariableOp2b
/model_4/batch_normalization_28/ReadVariableOp_1/model_4/batch_normalization_28/ReadVariableOp_12T
(model_4/conv2d_44/BiasAdd/ReadVariableOp(model_4/conv2d_44/BiasAdd/ReadVariableOp2R
'model_4/conv2d_44/Conv2D/ReadVariableOp'model_4/conv2d_44/Conv2D/ReadVariableOp2T
(model_4/conv2d_45/BiasAdd/ReadVariableOp(model_4/conv2d_45/BiasAdd/ReadVariableOp2R
'model_4/conv2d_45/Conv2D/ReadVariableOp'model_4/conv2d_45/Conv2D/ReadVariableOp2T
(model_4/conv2d_46/BiasAdd/ReadVariableOp(model_4/conv2d_46/BiasAdd/ReadVariableOp2R
'model_4/conv2d_46/Conv2D/ReadVariableOp'model_4/conv2d_46/Conv2D/ReadVariableOp2T
(model_4/conv2d_47/BiasAdd/ReadVariableOp(model_4/conv2d_47/BiasAdd/ReadVariableOp2R
'model_4/conv2d_47/Conv2D/ReadVariableOp'model_4/conv2d_47/Conv2D/ReadVariableOp2T
(model_4/conv2d_48/BiasAdd/ReadVariableOp(model_4/conv2d_48/BiasAdd/ReadVariableOp2R
'model_4/conv2d_48/Conv2D/ReadVariableOp'model_4/conv2d_48/Conv2D/ReadVariableOp2T
(model_4/conv2d_49/BiasAdd/ReadVariableOp(model_4/conv2d_49/BiasAdd/ReadVariableOp2R
'model_4/conv2d_49/Conv2D/ReadVariableOp'model_4/conv2d_49/Conv2D/ReadVariableOp2T
(model_4/conv2d_50/BiasAdd/ReadVariableOp(model_4/conv2d_50/BiasAdd/ReadVariableOp2R
'model_4/conv2d_50/Conv2D/ReadVariableOp'model_4/conv2d_50/Conv2D/ReadVariableOp2T
(model_4/conv2d_51/BiasAdd/ReadVariableOp(model_4/conv2d_51/BiasAdd/ReadVariableOp2R
'model_4/conv2d_51/Conv2D/ReadVariableOp'model_4/conv2d_51/Conv2D/ReadVariableOp2T
(model_4/conv2d_52/BiasAdd/ReadVariableOp(model_4/conv2d_52/BiasAdd/ReadVariableOp2R
'model_4/conv2d_52/Conv2D/ReadVariableOp'model_4/conv2d_52/Conv2D/ReadVariableOp2T
(model_4/conv2d_53/BiasAdd/ReadVariableOp(model_4/conv2d_53/BiasAdd/ReadVariableOp2R
'model_4/conv2d_53/Conv2D/ReadVariableOp'model_4/conv2d_53/Conv2D/ReadVariableOp2T
(model_4/conv2d_54/BiasAdd/ReadVariableOp(model_4/conv2d_54/BiasAdd/ReadVariableOp2R
'model_4/conv2d_54/Conv2D/ReadVariableOp'model_4/conv2d_54/Conv2D/ReadVariableOp2T
(model_4/conv2d_55/BiasAdd/ReadVariableOp(model_4/conv2d_55/BiasAdd/ReadVariableOp2R
'model_4/conv2d_55/Conv2D/ReadVariableOp'model_4/conv2d_55/Conv2D/ReadVariableOp:X T
/
_output_shapes
:���������			
!
_user_specified_name	input_5
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_241145

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_241345

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_21_layer_call_fn_243571

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_2402552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244581

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

*__inference_conv2d_44_layer_call_fn_243430

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_2410092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_240463

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
t
J__inference_concatenate_10_layer_call_and_return_conditional_losses_241906

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������		)2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������		)2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������		 :���������			:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_240328

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_45_layer_call_and_return_conditional_losses_241036

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�	
`
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_244714
x
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max/reduction_indices
MaxMaxxMax/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
Max\
subSubxMax:output:0*
T0*/
_output_shapes
:���������			2
subT
ExpExpsub:z:0*
T0*/
_output_shapes
:���������			2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������		*
	keep_dims(2
Sumn
truedivRealDivExp:y:0Sum:output:0*
T0*/
_output_shapes
:���������			2	
truedivg
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������			:R N
/
_output_shapes
:���������			

_user_specified_namex
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244349

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_28_layer_call_fn_244607

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2418632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_22_layer_call_fn_243783

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_2403592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_241163

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������		�:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:���������		�2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:���������		�::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
ޅ
�
C__inference_model_4_layer_call_and_return_conditional_losses_242498

inputs
conv2d_44_242362
conv2d_44_242364
conv2d_45_242367
conv2d_45_242369
conv2d_46_242372
conv2d_46_242374
conv2d_47_242377
conv2d_47_242379!
batch_normalization_21_242383!
batch_normalization_21_242385!
batch_normalization_21_242387!
batch_normalization_21_242389
conv2d_48_242392
conv2d_48_242394!
batch_normalization_22_242397!
batch_normalization_22_242399!
batch_normalization_22_242401!
batch_normalization_22_242403
conv2d_49_242406
conv2d_49_242408!
batch_normalization_23_242411!
batch_normalization_23_242413!
batch_normalization_23_242415!
batch_normalization_23_242417
conv2d_50_242420
conv2d_50_242422!
batch_normalization_24_242425!
batch_normalization_24_242427!
batch_normalization_24_242429!
batch_normalization_24_242431
conv2d_51_242434
conv2d_51_242436!
batch_normalization_25_242439!
batch_normalization_25_242441!
batch_normalization_25_242443!
batch_normalization_25_242445
conv2d_52_242448
conv2d_52_242450!
batch_normalization_26_242453!
batch_normalization_26_242455!
batch_normalization_26_242457!
batch_normalization_26_242459
conv2d_53_242462
conv2d_53_242464!
batch_normalization_27_242467!
batch_normalization_27_242469!
batch_normalization_27_242471!
batch_normalization_27_242473
conv2d_54_242476
conv2d_54_242478!
batch_normalization_28_242481!
batch_normalization_28_242483!
batch_normalization_28_242485!
batch_normalization_28_242487
conv2d_55_242491
conv2d_55_242493
identity��.batch_normalization_21/StatefulPartitionedCall�.batch_normalization_22/StatefulPartitionedCall�.batch_normalization_23/StatefulPartitionedCall�.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall�!conv2d_44/StatefulPartitionedCall�!conv2d_45/StatefulPartitionedCall�!conv2d_46/StatefulPartitionedCall�!conv2d_47/StatefulPartitionedCall�!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall�
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_44_242362conv2d_44_242364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_2410092#
!conv2d_44/StatefulPartitionedCall�
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45_242367conv2d_45_242369*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_2410362#
!conv2d_45/StatefulPartitionedCall�
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_46_242372conv2d_46_242374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_2410632#
!conv2d_46/StatefulPartitionedCall�
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_47_242377conv2d_47_242379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_2410902#
!conv2d_47/StatefulPartitionedCall�
concatenate_9/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*conv2d_45/StatefulPartitionedCall:output:0*conv2d_46/StatefulPartitionedCall:output:0*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_9_layer_call_and_return_conditional_losses_2411152
concatenate_9/PartitionedCall�
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0batch_normalization_21_242383batch_normalization_21_242385batch_normalization_21_242387batch_normalization_21_242389*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_24116320
.batch_normalization_21/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0conv2d_48_242392conv2d_48_242394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_48_layer_call_and_return_conditional_losses_2412102#
!conv2d_48/StatefulPartitionedCall�
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_22_242397batch_normalization_22_242399batch_normalization_22_242401batch_normalization_22_242403*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_24126320
.batch_normalization_22/StatefulPartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_49_242406conv2d_49_242408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_49_layer_call_and_return_conditional_losses_2413102#
!conv2d_49/StatefulPartitionedCall�
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_23_242411batch_normalization_23_242413batch_normalization_23_242415batch_normalization_23_242417*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_24136320
.batch_normalization_23/StatefulPartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_50_242420conv2d_50_242422*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_2414102#
!conv2d_50/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_24_242425batch_normalization_24_242427batch_normalization_24_242429batch_normalization_24_242431*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_24146320
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_51_242434conv2d_51_242436*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2415102#
!conv2d_51/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_25_242439batch_normalization_25_242441batch_normalization_25_242443batch_normalization_25_242445*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_24156320
.batch_normalization_25/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0conv2d_52_242448conv2d_52_242450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2416102#
!conv2d_52/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_26_242453batch_normalization_26_242455batch_normalization_26_242457batch_normalization_26_242459*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_24166320
.batch_normalization_26/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_53_242462conv2d_53_242464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2417102#
!conv2d_53/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_27_242467batch_normalization_27_242469batch_normalization_27_242471batch_normalization_27_242473*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_24176320
.batch_normalization_27/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_54_242476conv2d_54_242478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_2418102#
!conv2d_54/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_28_242481batch_normalization_28_242483batch_normalization_28_242485batch_normalization_28_242487*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_24186320
.batch_normalization_28/StatefulPartitionedCall�
concatenate_10/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_10_layer_call_and_return_conditional_losses_2419062 
concatenate_10/PartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_55_242491conv2d_55_242493*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_2419252#
!conv2d_55/StatefulPartitionedCall�
softmax_map_4/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_2419522
softmax_map_4/PartitionedCall�
IdentityIdentity&softmax_map_4/PartitionedCall:output:0/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�

�
E__inference_conv2d_50_layer_call_and_return_conditional_losses_243942

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_46_layer_call_and_return_conditional_losses_243461

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_25_layer_call_fn_244150

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_2406402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_240432

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

*__inference_conv2d_54_layer_call_fn_244543

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_2418102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_241445

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_48_layer_call_and_return_conditional_losses_243646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:		�@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������		�::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
[
/__inference_concatenate_10_layer_call_fn_244684
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_10_layer_call_and_return_conditional_losses_2419062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������		)2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������		 :���������			:Y U
/
_output_shapes
:���������		 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������			
"
_user_specified_name
inputs/1
�
�
7__inference_batch_normalization_28_layer_call_fn_244658

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2409522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_240879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244497

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_240359

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_52_layer_call_and_return_conditional_losses_241610

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_46_layer_call_and_return_conditional_losses_241063

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_22_layer_call_fn_243706

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_2412452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

*__inference_conv2d_46_layer_call_fn_243470

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_2410632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244183

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_22_layer_call_fn_243770

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_2403282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243757

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_25_layer_call_fn_244227

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_2415632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�

�
E__inference_conv2d_50_layer_call_and_return_conditional_losses_241410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������		@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_241563

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244035

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_21_layer_call_fn_243558

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_2402242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243841

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������		@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������		@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_47_layer_call_and_return_conditional_losses_241090

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:			 *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:���������		 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������			::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_27_layer_call_fn_244523

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2417632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������		 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������		 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		 
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_243410

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������			*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_2424982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������			2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������			::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������			
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_58
serving_default_input_5:0���������			I
softmax_map_48
StatefulPartitionedCall:0���������			tensorflow/serving/predict:��
څ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
layer_with_weights-18
layer-20
layer-21
layer_with_weights-19
layer-22
layer-23
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�}
_tf_keras_network�}{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_44", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_45", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_45", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_46", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_47", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["conv2d_44", 0, 0, {}], ["conv2d_45", 0, 0, {}], ["conv2d_46", 0, 0, {}], ["conv2d_47", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_48", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["conv2d_48", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_49", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_49", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_50", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_24", "inbound_nodes": [[["conv2d_50", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_51", "inbound_nodes": [[["batch_normalization_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_25", "inbound_nodes": [[["conv2d_51", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_52", "inbound_nodes": [[["batch_normalization_25", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_26", "inbound_nodes": [[["conv2d_52", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_53", "inbound_nodes": [[["batch_normalization_26", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["conv2d_53", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_54", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_28", "inbound_nodes": [[["conv2d_54", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["batch_normalization_28", 0, 0, {}], ["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_55", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "SoftmaxMap", "config": {"layer was saved without config": true}, "name": "softmax_map_4", "inbound_nodes": [[["conv2d_55", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["softmax_map_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9, 9, 9]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9, 9, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
�	

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_44", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}}
�	

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_45", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}}
�	

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}}
�	

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}}
�
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 9, 9, 32]}, {"class_name": "TensorShape", "items": [null, 9, 9, 32]}, {"class_name": "TensorShape", "items": [null, 9, 9, 32]}, {"class_name": "TensorShape", "items": [null, 9, 9, 32]}]}
�	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 128]}}
�	

Dkernel
Ebias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 128]}}
�	
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^trainable_variables
_regularization_losses
`	variables
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	

bkernel
cbias
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	

qkernel
rbias
strainable_variables
tregularization_losses
u	variables
v	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�	
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_54", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 32]}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 9, 9, 32]}, {"class_name": "TensorShape", "items": [null, 9, 9, 9]}]}
�	
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 41}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 41]}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SoftmaxMap", "name": "softmax_map_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 9]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem� m�%m�&m�+m�,m�1m�2m�<m�=m�Dm�Em�Km�Lm�Sm�Tm�Zm�[m�bm�cm�im�jm�qm�rm�xm�ym�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v� v�%v�&v�+v�,v�1v�2v�<v�=v�Dv�Ev�Kv�Lv�Sv�Tv�Zv�[v�bv�cv�iv�jv�qv�rv�xv�yv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
0
 1
%2
&3
+4
,5
16
27
<8
=9
D10
E11
K12
L13
S14
T15
Z16
[17
b18
c19
i20
j21
q22
r23
x24
y25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
 1
%2
&3
+4
,5
16
27
<8
=9
>10
?11
D12
E13
K14
L15
M16
N17
S18
T19
Z20
[21
\22
]23
b24
c25
i26
j27
k28
l29
q30
r31
x32
y33
z34
{35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55"
trackable_list_wrapper
�
trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
*:(	 2conv2d_44/kernel
: 2conv2d_44/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
!trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
"regularization_losses
�metrics
#	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(		 2conv2d_45/kernel
: 2conv2d_45/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
'trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
(regularization_losses
�metrics
)	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(		 2conv2d_46/kernel
: 2conv2d_46/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
-trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
.regularization_losses
�metrics
/	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(			 2conv2d_47/kernel
: 2conv2d_47/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
3trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
4regularization_losses
�metrics
5	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
7trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
8regularization_losses
�metrics
9	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_21/gamma
*:(�2batch_normalization_21/beta
3:1� (2"batch_normalization_21/moving_mean
7:5� (2&batch_normalization_21/moving_variance
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
�
@trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Aregularization_losses
�metrics
B	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)		�@2conv2d_48/kernel
:@2conv2d_48/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
�
Ftrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Gregularization_losses
�metrics
H	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_22/gamma
):'@2batch_normalization_22/beta
2:0@ (2"batch_normalization_22/moving_mean
6:4@ (2&batch_normalization_22/moving_variance
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
�
Otrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Pregularization_losses
�metrics
Q	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_49/kernel
:@2conv2d_49/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
Utrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
Vregularization_losses
�metrics
W	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_23/gamma
):'@2batch_normalization_23/beta
2:0@ (2"batch_normalization_23/moving_mean
6:4@ (2&batch_normalization_23/moving_variance
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
Z0
[1
\2
]3"
trackable_list_wrapper
�
^trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
_regularization_losses
�metrics
`	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_50/kernel
:@2conv2d_50/bias
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
�
dtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
eregularization_losses
�metrics
f	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_24/gamma
):'@2batch_normalization_24/beta
2:0@ (2"batch_normalization_24/moving_mean
6:4@ (2&batch_normalization_24/moving_variance
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
i0
j1
k2
l3"
trackable_list_wrapper
�
mtrainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
nregularization_losses
�metrics
o	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@ 2conv2d_51/kernel
: 2conv2d_51/bias
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
�
strainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
tregularization_losses
�metrics
u	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_25/gamma
):' 2batch_normalization_25/beta
2:0  (2"batch_normalization_25/moving_mean
6:4  (2&batch_normalization_25/moving_variance
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
�
|trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
}regularization_losses
�metrics
~	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_52/kernel
: 2conv2d_52/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_26/gamma
):' 2batch_normalization_26/beta
2:0  (2"batch_normalization_26/moving_mean
6:4  (2&batch_normalization_26/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_53/kernel
: 2conv2d_53/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_27/gamma
):' 2batch_normalization_27/beta
2:0  (2"batch_normalization_27/moving_mean
6:4  (2&batch_normalization_27/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_54/kernel
: 2conv2d_54/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_28/gamma
):' 2batch_normalization_28/beta
2:0  (2"batch_normalization_28/moving_mean
6:4  (2&batch_normalization_28/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:()	2conv2d_55/kernel
:	2conv2d_55/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
 �layer_regularization_losses
�layers
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
>0
?1
M2
N3
\4
]5
k6
l7
z8
{9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
/:-	 2Adam/conv2d_44/kernel/m
!: 2Adam/conv2d_44/bias/m
/:-		 2Adam/conv2d_45/kernel/m
!: 2Adam/conv2d_45/bias/m
/:-		 2Adam/conv2d_46/kernel/m
!: 2Adam/conv2d_46/bias/m
/:-			 2Adam/conv2d_47/kernel/m
!: 2Adam/conv2d_47/bias/m
0:.�2#Adam/batch_normalization_21/gamma/m
/:-�2"Adam/batch_normalization_21/beta/m
0:.		�@2Adam/conv2d_48/kernel/m
!:@2Adam/conv2d_48/bias/m
/:-@2#Adam/batch_normalization_22/gamma/m
.:,@2"Adam/batch_normalization_22/beta/m
/:-@@2Adam/conv2d_49/kernel/m
!:@2Adam/conv2d_49/bias/m
/:-@2#Adam/batch_normalization_23/gamma/m
.:,@2"Adam/batch_normalization_23/beta/m
/:-@@2Adam/conv2d_50/kernel/m
!:@2Adam/conv2d_50/bias/m
/:-@2#Adam/batch_normalization_24/gamma/m
.:,@2"Adam/batch_normalization_24/beta/m
/:-@ 2Adam/conv2d_51/kernel/m
!: 2Adam/conv2d_51/bias/m
/:- 2#Adam/batch_normalization_25/gamma/m
.:, 2"Adam/batch_normalization_25/beta/m
/:-  2Adam/conv2d_52/kernel/m
!: 2Adam/conv2d_52/bias/m
/:- 2#Adam/batch_normalization_26/gamma/m
.:, 2"Adam/batch_normalization_26/beta/m
/:-  2Adam/conv2d_53/kernel/m
!: 2Adam/conv2d_53/bias/m
/:- 2#Adam/batch_normalization_27/gamma/m
.:, 2"Adam/batch_normalization_27/beta/m
/:-  2Adam/conv2d_54/kernel/m
!: 2Adam/conv2d_54/bias/m
/:- 2#Adam/batch_normalization_28/gamma/m
.:, 2"Adam/batch_normalization_28/beta/m
/:-)	2Adam/conv2d_55/kernel/m
!:	2Adam/conv2d_55/bias/m
/:-	 2Adam/conv2d_44/kernel/v
!: 2Adam/conv2d_44/bias/v
/:-		 2Adam/conv2d_45/kernel/v
!: 2Adam/conv2d_45/bias/v
/:-		 2Adam/conv2d_46/kernel/v
!: 2Adam/conv2d_46/bias/v
/:-			 2Adam/conv2d_47/kernel/v
!: 2Adam/conv2d_47/bias/v
0:.�2#Adam/batch_normalization_21/gamma/v
/:-�2"Adam/batch_normalization_21/beta/v
0:.		�@2Adam/conv2d_48/kernel/v
!:@2Adam/conv2d_48/bias/v
/:-@2#Adam/batch_normalization_22/gamma/v
.:,@2"Adam/batch_normalization_22/beta/v
/:-@@2Adam/conv2d_49/kernel/v
!:@2Adam/conv2d_49/bias/v
/:-@2#Adam/batch_normalization_23/gamma/v
.:,@2"Adam/batch_normalization_23/beta/v
/:-@@2Adam/conv2d_50/kernel/v
!:@2Adam/conv2d_50/bias/v
/:-@2#Adam/batch_normalization_24/gamma/v
.:,@2"Adam/batch_normalization_24/beta/v
/:-@ 2Adam/conv2d_51/kernel/v
!: 2Adam/conv2d_51/bias/v
/:- 2#Adam/batch_normalization_25/gamma/v
.:, 2"Adam/batch_normalization_25/beta/v
/:-  2Adam/conv2d_52/kernel/v
!: 2Adam/conv2d_52/bias/v
/:- 2#Adam/batch_normalization_26/gamma/v
.:, 2"Adam/batch_normalization_26/beta/v
/:-  2Adam/conv2d_53/kernel/v
!: 2Adam/conv2d_53/bias/v
/:- 2#Adam/batch_normalization_27/gamma/v
.:, 2"Adam/batch_normalization_27/beta/v
/:-  2Adam/conv2d_54/kernel/v
!: 2Adam/conv2d_54/bias/v
/:- 2#Adam/batch_normalization_28/gamma/v
.:, 2"Adam/batch_normalization_28/beta/v
/:-)	2Adam/conv2d_55/kernel/v
!:	2Adam/conv2d_55/bias/v
�2�
C__inference_model_4_layer_call_and_return_conditional_losses_242966
C__inference_model_4_layer_call_and_return_conditional_losses_243176
C__inference_model_4_layer_call_and_return_conditional_losses_241961
C__inference_model_4_layer_call_and_return_conditional_losses_242100�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_240162�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_5���������			
�2�
(__inference_model_4_layer_call_fn_242357
(__inference_model_4_layer_call_fn_242613
(__inference_model_4_layer_call_fn_243410
(__inference_model_4_layer_call_fn_243293�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_44_layer_call_and_return_conditional_losses_243421�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_44_layer_call_fn_243430�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_45_layer_call_and_return_conditional_losses_243441�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_45_layer_call_fn_243450�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_46_layer_call_and_return_conditional_losses_243461�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_46_layer_call_fn_243470�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_47_layer_call_and_return_conditional_losses_243481�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_47_layer_call_fn_243490�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_concatenate_9_layer_call_and_return_conditional_losses_243499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_concatenate_9_layer_call_fn_243507�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243591
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243527
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243609
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243545�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_21_layer_call_fn_243622
7__inference_batch_normalization_21_layer_call_fn_243571
7__inference_batch_normalization_21_layer_call_fn_243635
7__inference_batch_normalization_21_layer_call_fn_243558�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_48_layer_call_and_return_conditional_losses_243646�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_48_layer_call_fn_243655�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243693
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243757
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243739
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243675�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_22_layer_call_fn_243706
7__inference_batch_normalization_22_layer_call_fn_243783
7__inference_batch_normalization_22_layer_call_fn_243719
7__inference_batch_normalization_22_layer_call_fn_243770�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_49_layer_call_and_return_conditional_losses_243794�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_49_layer_call_fn_243803�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243841
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243823
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243887
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243905�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_23_layer_call_fn_243931
7__inference_batch_normalization_23_layer_call_fn_243854
7__inference_batch_normalization_23_layer_call_fn_243918
7__inference_batch_normalization_23_layer_call_fn_243867�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_50_layer_call_and_return_conditional_losses_243942�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_50_layer_call_fn_243951�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244035
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244053
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243989
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243971�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_24_layer_call_fn_244066
7__inference_batch_normalization_24_layer_call_fn_244015
7__inference_batch_normalization_24_layer_call_fn_244002
7__inference_batch_normalization_24_layer_call_fn_244079�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_51_layer_call_and_return_conditional_losses_244090�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_51_layer_call_fn_244099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244137
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244119
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244183
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244201�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_25_layer_call_fn_244227
7__inference_batch_normalization_25_layer_call_fn_244214
7__inference_batch_normalization_25_layer_call_fn_244150
7__inference_batch_normalization_25_layer_call_fn_244163�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_52_layer_call_and_return_conditional_losses_244238�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_52_layer_call_fn_244247�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244349
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244285
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244267
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244331�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_26_layer_call_fn_244298
7__inference_batch_normalization_26_layer_call_fn_244362
7__inference_batch_normalization_26_layer_call_fn_244375
7__inference_batch_normalization_26_layer_call_fn_244311�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_53_layer_call_and_return_conditional_losses_244386�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_53_layer_call_fn_244395�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244433
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244415
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244497
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244479�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_27_layer_call_fn_244510
7__inference_batch_normalization_27_layer_call_fn_244459
7__inference_batch_normalization_27_layer_call_fn_244446
7__inference_batch_normalization_27_layer_call_fn_244523�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_54_layer_call_and_return_conditional_losses_244534�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_54_layer_call_fn_244543�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244581
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244645
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244563
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244627�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_28_layer_call_fn_244671
7__inference_batch_normalization_28_layer_call_fn_244658
7__inference_batch_normalization_28_layer_call_fn_244594
7__inference_batch_normalization_28_layer_call_fn_244607�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_concatenate_10_layer_call_and_return_conditional_losses_244678�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_concatenate_10_layer_call_fn_244684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_55_layer_call_and_return_conditional_losses_244694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_55_layer_call_fn_244703�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_244714�
���
FullArgSpec 
args�
jself
jx
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_softmax_map_4_layer_call_fn_244719�
���
FullArgSpec 
args�
jself
jx
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_242740input_5"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_240162�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������8�5
.�+
)�&
input_5���������			
� "E�B
@
softmax_map_4/�,
softmax_map_4���������			�
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243527�<=>?N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243545�<=>?N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243591t<=>?<�9
2�/
)�&
inputs���������		�
p
� ".�+
$�!
0���������		�
� �
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_243609t<=>?<�9
2�/
)�&
inputs���������		�
p 
� ".�+
$�!
0���������		�
� �
7__inference_batch_normalization_21_layer_call_fn_243558�<=>?N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_21_layer_call_fn_243571�<=>?N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_21_layer_call_fn_243622g<=>?<�9
2�/
)�&
inputs���������		�
p
� "!����������		��
7__inference_batch_normalization_21_layer_call_fn_243635g<=>?<�9
2�/
)�&
inputs���������		�
p 
� "!����������		��
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243675rKLMN;�8
1�.
(�%
inputs���������		@
p
� "-�*
#� 
0���������		@
� �
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243693rKLMN;�8
1�.
(�%
inputs���������		@
p 
� "-�*
#� 
0���������		@
� �
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243739�KLMNM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_243757�KLMNM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_22_layer_call_fn_243706eKLMN;�8
1�.
(�%
inputs���������		@
p
� " ����������		@�
7__inference_batch_normalization_22_layer_call_fn_243719eKLMN;�8
1�.
(�%
inputs���������		@
p 
� " ����������		@�
7__inference_batch_normalization_22_layer_call_fn_243770�KLMNM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_22_layer_call_fn_243783�KLMNM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243823rZ[\];�8
1�.
(�%
inputs���������		@
p
� "-�*
#� 
0���������		@
� �
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243841rZ[\];�8
1�.
(�%
inputs���������		@
p 
� "-�*
#� 
0���������		@
� �
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243887�Z[\]M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_243905�Z[\]M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_23_layer_call_fn_243854eZ[\];�8
1�.
(�%
inputs���������		@
p
� " ����������		@�
7__inference_batch_normalization_23_layer_call_fn_243867eZ[\];�8
1�.
(�%
inputs���������		@
p 
� " ����������		@�
7__inference_batch_normalization_23_layer_call_fn_243918�Z[\]M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_23_layer_call_fn_243931�Z[\]M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243971�ijklM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_243989�ijklM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244035rijkl;�8
1�.
(�%
inputs���������		@
p
� "-�*
#� 
0���������		@
� �
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_244053rijkl;�8
1�.
(�%
inputs���������		@
p 
� "-�*
#� 
0���������		@
� �
7__inference_batch_normalization_24_layer_call_fn_244002�ijklM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_24_layer_call_fn_244015�ijklM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_24_layer_call_fn_244066eijkl;�8
1�.
(�%
inputs���������		@
p
� " ����������		@�
7__inference_batch_normalization_24_layer_call_fn_244079eijkl;�8
1�.
(�%
inputs���������		@
p 
� " ����������		@�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244119�xyz{M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244137�xyz{M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244183rxyz{;�8
1�.
(�%
inputs���������		 
p
� "-�*
#� 
0���������		 
� �
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_244201rxyz{;�8
1�.
(�%
inputs���������		 
p 
� "-�*
#� 
0���������		 
� �
7__inference_batch_normalization_25_layer_call_fn_244150�xyz{M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_25_layer_call_fn_244163�xyz{M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_25_layer_call_fn_244214exyz{;�8
1�.
(�%
inputs���������		 
p
� " ����������		 �
7__inference_batch_normalization_25_layer_call_fn_244227exyz{;�8
1�.
(�%
inputs���������		 
p 
� " ����������		 �
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244267�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244285�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244331v����;�8
1�.
(�%
inputs���������		 
p
� "-�*
#� 
0���������		 
� �
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_244349v����;�8
1�.
(�%
inputs���������		 
p 
� "-�*
#� 
0���������		 
� �
7__inference_batch_normalization_26_layer_call_fn_244298�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_26_layer_call_fn_244311�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_26_layer_call_fn_244362i����;�8
1�.
(�%
inputs���������		 
p
� " ����������		 �
7__inference_batch_normalization_26_layer_call_fn_244375i����;�8
1�.
(�%
inputs���������		 
p 
� " ����������		 �
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244415�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244433�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244479v����;�8
1�.
(�%
inputs���������		 
p
� "-�*
#� 
0���������		 
� �
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_244497v����;�8
1�.
(�%
inputs���������		 
p 
� "-�*
#� 
0���������		 
� �
7__inference_batch_normalization_27_layer_call_fn_244446�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_27_layer_call_fn_244459�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_27_layer_call_fn_244510i����;�8
1�.
(�%
inputs���������		 
p
� " ����������		 �
7__inference_batch_normalization_27_layer_call_fn_244523i����;�8
1�.
(�%
inputs���������		 
p 
� " ����������		 �
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244563v����;�8
1�.
(�%
inputs���������		 
p
� "-�*
#� 
0���������		 
� �
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244581v����;�8
1�.
(�%
inputs���������		 
p 
� "-�*
#� 
0���������		 
� �
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244627�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_244645�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
7__inference_batch_normalization_28_layer_call_fn_244594i����;�8
1�.
(�%
inputs���������		 
p
� " ����������		 �
7__inference_batch_normalization_28_layer_call_fn_244607i����;�8
1�.
(�%
inputs���������		 
p 
� " ����������		 �
7__inference_batch_normalization_28_layer_call_fn_244658�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_28_layer_call_fn_244671�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
J__inference_concatenate_10_layer_call_and_return_conditional_losses_244678�j�g
`�]
[�X
*�'
inputs/0���������		 
*�'
inputs/1���������			
� "-�*
#� 
0���������		)
� �
/__inference_concatenate_10_layer_call_fn_244684�j�g
`�]
[�X
*�'
inputs/0���������		 
*�'
inputs/1���������			
� " ����������		)�
I__inference_concatenate_9_layer_call_and_return_conditional_losses_243499����
���
���
*�'
inputs/0���������		 
*�'
inputs/1���������		 
*�'
inputs/2���������		 
*�'
inputs/3���������		 
� ".�+
$�!
0���������		�
� �
.__inference_concatenate_9_layer_call_fn_243507����
���
���
*�'
inputs/0���������		 
*�'
inputs/1���������		 
*�'
inputs/2���������		 
*�'
inputs/3���������		 
� "!����������		��
E__inference_conv2d_44_layer_call_and_return_conditional_losses_243421l 7�4
-�*
(�%
inputs���������			
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_44_layer_call_fn_243430_ 7�4
-�*
(�%
inputs���������			
� " ����������		 �
E__inference_conv2d_45_layer_call_and_return_conditional_losses_243441l%&7�4
-�*
(�%
inputs���������			
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_45_layer_call_fn_243450_%&7�4
-�*
(�%
inputs���������			
� " ����������		 �
E__inference_conv2d_46_layer_call_and_return_conditional_losses_243461l+,7�4
-�*
(�%
inputs���������			
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_46_layer_call_fn_243470_+,7�4
-�*
(�%
inputs���������			
� " ����������		 �
E__inference_conv2d_47_layer_call_and_return_conditional_losses_243481l127�4
-�*
(�%
inputs���������			
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_47_layer_call_fn_243490_127�4
-�*
(�%
inputs���������			
� " ����������		 �
E__inference_conv2d_48_layer_call_and_return_conditional_losses_243646mDE8�5
.�+
)�&
inputs���������		�
� "-�*
#� 
0���������		@
� �
*__inference_conv2d_48_layer_call_fn_243655`DE8�5
.�+
)�&
inputs���������		�
� " ����������		@�
E__inference_conv2d_49_layer_call_and_return_conditional_losses_243794lST7�4
-�*
(�%
inputs���������		@
� "-�*
#� 
0���������		@
� �
*__inference_conv2d_49_layer_call_fn_243803_ST7�4
-�*
(�%
inputs���������		@
� " ����������		@�
E__inference_conv2d_50_layer_call_and_return_conditional_losses_243942lbc7�4
-�*
(�%
inputs���������		@
� "-�*
#� 
0���������		@
� �
*__inference_conv2d_50_layer_call_fn_243951_bc7�4
-�*
(�%
inputs���������		@
� " ����������		@�
E__inference_conv2d_51_layer_call_and_return_conditional_losses_244090lqr7�4
-�*
(�%
inputs���������		@
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_51_layer_call_fn_244099_qr7�4
-�*
(�%
inputs���������		@
� " ����������		 �
E__inference_conv2d_52_layer_call_and_return_conditional_losses_244238n��7�4
-�*
(�%
inputs���������		 
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_52_layer_call_fn_244247a��7�4
-�*
(�%
inputs���������		 
� " ����������		 �
E__inference_conv2d_53_layer_call_and_return_conditional_losses_244386n��7�4
-�*
(�%
inputs���������		 
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_53_layer_call_fn_244395a��7�4
-�*
(�%
inputs���������		 
� " ����������		 �
E__inference_conv2d_54_layer_call_and_return_conditional_losses_244534n��7�4
-�*
(�%
inputs���������		 
� "-�*
#� 
0���������		 
� �
*__inference_conv2d_54_layer_call_fn_244543a��7�4
-�*
(�%
inputs���������		 
� " ����������		 �
E__inference_conv2d_55_layer_call_and_return_conditional_losses_244694n��7�4
-�*
(�%
inputs���������		)
� "-�*
#� 
0���������			
� �
*__inference_conv2d_55_layer_call_fn_244703a��7�4
-�*
(�%
inputs���������		)
� " ����������			�
C__inference_model_4_layer_call_and_return_conditional_losses_241961�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������@�=
6�3
)�&
input_5���������			
p

 
� "-�*
#� 
0���������			
� �
C__inference_model_4_layer_call_and_return_conditional_losses_242100�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������@�=
6�3
)�&
input_5���������			
p 

 
� "-�*
#� 
0���������			
� �
C__inference_model_4_layer_call_and_return_conditional_losses_242966�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������?�<
5�2
(�%
inputs���������			
p

 
� "-�*
#� 
0���������			
� �
C__inference_model_4_layer_call_and_return_conditional_losses_243176�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������?�<
5�2
(�%
inputs���������			
p 

 
� "-�*
#� 
0���������			
� �
(__inference_model_4_layer_call_fn_242357�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������@�=
6�3
)�&
input_5���������			
p

 
� " ����������			�
(__inference_model_4_layer_call_fn_242613�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������@�=
6�3
)�&
input_5���������			
p 

 
� " ����������			�
(__inference_model_4_layer_call_fn_243293�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������?�<
5�2
(�%
inputs���������			
p

 
� " ����������			�
(__inference_model_4_layer_call_fn_243410�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������?�<
5�2
(�%
inputs���������			
p 

 
� " ����������			�
$__inference_signature_wrapper_242740�L %&+,12<=>?DEKLMNSTZ[\]bcijklqrxyz{��������������������C�@
� 
9�6
4
input_5)�&
input_5���������			"E�B
@
softmax_map_4/�,
softmax_map_4���������			�
I__inference_softmax_map_4_layer_call_and_return_conditional_losses_244714g6�3
,�)
#� 
x���������			

 
� "-�*
#� 
0���������			
� �
.__inference_softmax_map_4_layer_call_fn_244719Z6�3
,�)
#� 
x���������			

 
� " ����������			