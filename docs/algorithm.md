# 算法：冻结编辑器与范围投影

该流程将图像编辑分为两个相接的步骤：冻结的指令编辑器生成候选图，范围投影把候选图中的改动限制到人工标注的材质区域。编辑器承担语义与形体重组，投影承担范围保持。

## 输入与坐标

原图 $x$ 和候选图 $r$ 是同尺寸的 RGB 图像，像素值取自 $\{0,\ldots,255\}$。二值掩码 $M$ 与图像具有相同的宽高；$M(p)=1$ 表示像素 $p$ 允许编辑，$M(p)=0$ 表示该像素必须保留。材质区域通过手工 polygon 或 bitmap 标注得到，配置中的 `allow`、`protect` 与可选 `base_mask` 决定去除保护区后的完整允许域 $M$。

`active_rects` 可进一步选择本轮编辑区域。设这些矩形的并集为 $R$，本轮有效区域为

$$
A=M\cap R.
$$

未指定本轮矩形时，取 $A=M$。因此总有 $A\subseteq M$。羽化针对 $A$ 计算，范围审计针对 $M$ 检查。

配置的 `coordinate_size` 必须匹配原图尺寸。图像尺寸、裁剪位置或对象位置改变后，原来的 polygon 和 bitmap 不会自动对应新区域。先核对掩码与图像对齐，再执行投影。

候选图可来自已有文件，也可由可选 OpenAI 或 Klein 后端生成。输入图像读取后统一为 RGB，透明图像需先显式铺底。投影只依赖候选图的像素，不需要访问编辑器的内部状态。

## 掩码配置

合成示例的 `scope.json` 展示了 polygon 标注方式。各配置项的含义如下：

| 配置项 | 格式与作用 |
| --- | --- |
| `coordinate_size` | `[width, height]`，用于检查标注尺寸 |
| `allow` | 多边形列表，每项的 `points` 是至少三个 `[x, y]` 顶点；区域取并集 |
| `protect` | 多边形列表，每项使用 `points`；从允许域中减去这些区域 |
| `base_mask` | 可选 bitmap 路径，相对配置文件所在目录；灰度值大于 0 的像素加入允许域 |
| `active_rects` | 可选整数矩形列表，每项为 `[left, top, right, bottom]`，限定本轮区域 |
| `feather_px` | 有限非负数，默认 7；为 0 时执行硬替换 |

矩形采用半开坐标：$[L,R)\times[T,B)$，所以宽高为 $R-L$ 与 $B-T$。`active_rects` 的矩形先取并集，再与允许域求交。省略该项时使用整个允许域，空列表则不激活任何像素。

`allow` 中每个多边形还可以设置 `color_guard`，按原图颜色进一步筛选该多边形内部的像素。支持的阈值为三通道 `min_rgb`、`max_rgb` 与 `b_minus_r_min`。该筛选是确定性的颜色条件，材质区域仍来自人工标注。

## 从掩码生成羽化权重

实现先在 $A$ 四周补一圈零像素，再用欧氏距离变换计算 $d(p)$。这里的距离是像素中心到最近零像素中心的距离；相邻背景的有效像素距离为 1，图像外部也视为背景。令 $f=\texttt{feather\_px}$ 为羽化宽度，单位为像素。对 $f>0$，权重定义为

$$
\alpha(p)=
\begin{cases}
\min\!\left(1,\max\!\left(0,\dfrac{d(p)}{f}\right)\right), & A(p)=1,\\
0, & A(p)=0.
\end{cases}
$$

边界附近的 $\alpha$ 较小，原图占比更高；当 $d(p)\ge f$ 时，$\alpha(p)=1$，该位置采用候选图。权重始终属于 $[0,1]$。因为 $A\subseteq M$，完整允许域外的权重也为零。

当 $f=0$ 时，有效区域的权重直接取 1，执行硬替换。生成的 `alpha.png` 使用 8 位灰度保存，先计算 $q(p)=\operatorname{round}(255\alpha(p))$，再读取为实际权重

$$
\widehat\alpha(p)=\dfrac{q(p)}{255},
\qquad q(p)\in\{0,\ldots,255\}.
$$

后续公式中的 $\alpha$ 均指实际使用的权重 $\widehat\alpha$。保留该 PNG 可以避免重新计算羽化时的量化差异。省略 `project --alpha` 时，投影取 $\alpha=M$。

## 逐像素投影与整数输出

先对每个像素、每个颜色通道计算连续值

$$
z(p)=x(p)+\alpha(p)\bigl(r(p)-x(p)\bigr)
     =\bigl(1-\alpha(p)\bigr)x(p)+\alpha(p)r(p).
$$

两个系数均非负且和为一，所以当输入值位于 $[0,255]$ 时，$z(p)$ 也位于该区间。输出实现仍显式裁剪并舍入，得到 8 位整数图像：

$$
\widetilde y(p)=\operatorname{round}\!\left(
\operatorname{clip}\bigl(z(p),0,255\bigr)
\right).
$$

保存前，按二值掩码重新写回范围外的原图像素：

$$
y(p)=
\begin{cases}
\widetilde y(p), & M(p)=1,\\
x(p), & M(p)=0.
\end{cases}
$$

显式恢复使范围外的保持条件直接体现在输出数组中。使用无损图像格式保存并读取后，可继续逐值验证该条件。

实现以 `float32` 计算混合值，`round` 使用 NumPy `rint` 的规则，即恰好位于两个整数中间时舍入到偶数。最终保存为 8 位 RGB PNG。

## 范围保持的推导

对任意 $M(p)=0$ 的像素，定义给出 $\alpha(p)=0$，于是

$$
z(p)=x(p)+0\cdot\bigl(r(p)-x(p)\bigr)=x(p).
$$

由于 $x(p)$ 已经是 $[0,255]$ 内的整数，裁剪与舍入保持其值；最终恢复操作再次给出 $y(p)=x(p)$。因此，在解码后的同尺寸 RGB 像素数组上，

$$
\max_{p:\,M(p)=0}\lVert y(p)-x(p)\rVert_\infty=0,
\qquad
\#\{p:M(p)=0,\ y(p)\ne x(p)\}=0.
$$

审计报告将这两项分别记为 `protected_max_rgb_error` 与 `protected_changed_pixels`；均为零时，`scope_preservation_passed` 为 `true`。若允许域覆盖全图，受保护像素数为零，两项误差也记为零。它们检验允许域外是否改变，不评价范围内的视觉效果。

## 命令与检查

合成示例先生成原图、候选图与掩码配置：

```powershell
python examples/make_demo.py --output-dir demo_data
aigc-clear mask --input demo_data/source.png --config demo_data/scope.json --output-dir demo_data/masks
```

`mask` 输出下列文件：

| 文件 | 含义 |
| --- | --- |
| `mask.png` | 完整允许域 $M$ |
| `active_mask.png` | 本轮有效区域 $A$ |
| `alpha.png` | 量化后的距离羽化权重 |
| `overlay.png` | 掩码可视化，用于核对标注位置 |
| `mask_report.json` | 掩码统计报告 |

再投影并生成报告：

```powershell
aigc-clear project --input demo_data/source.png --proposal demo_data/proposal.png --mask demo_data/masks/mask.png --alpha demo_data/masks/alpha.png --output demo_data/result.png --report demo_data/report.json
```

最后独立审计结果，并查看并排图：

```powershell
aigc-clear audit --reference demo_data/source.png --image demo_data/result.png --mask demo_data/masks/mask.png
aigc-clear compare --images demo_data/source.png demo_data/result.png --output demo_data/comparison.png
```

`run` 将掩码生成、投影、审计与并排预览合为一次调用。已有候选图时使用：

```powershell
aigc-clear run --input demo_data/source.png --config demo_data/scope.json --proposal demo_data/proposal.png --output-dir demo_data/run
```

调用编辑器时，用 `--backend openai` 或 `--backend klein` 替代 `--proposal`，并通过 `--prompt-file` 或 `--prompt` 提供编辑指令。

`propose` 与调用后端的 `run` 均支持 `--crop L T R B`。裁剪使用半开坐标，必须位于原图内部。编辑器返回与裁剪同尺寸的候选图后，程序将其按原坐标回填到原图副本，得到原尺寸的 proposal；裁剪外保持原图像素。输出尺寸不匹配时，程序保留 `.raw.png` 并停止，不自动缩放。

`compare` 会缩小图像生成查看用的并排预览，审计始终使用原尺寸图像。需要中文标签时，通过 `--labels` 设置文字，并用 `--font` 指定支持中文的字体文件。

写文件命令默认拒绝覆盖，确认需要更新时使用 `--overwrite`。投影报告记录输入、候选图、掩码、alpha 与结果的 SHA-256，以及范围审计指标。保留这些图像文件与报告，即可重复执行与核对投影。

## 编辑器与结果来源

原始成功路径使用内置 `imagegen` 生成候选图，再用人工材质掩码完成范围投影。该生成过程的模型 checkpoint 与随机种子没有公开，因此文本指令本身不足以复现原始候选图。本仓库提供已有候选图、OpenAI SDK `images.edit` 和 Klein 官方 revision 三种入口；后两者可用于开展新的编辑实验。

采用公开 Klein 模型的代理实验在原生 512 裁剪上检查了 `sidehair2`、`sidehair3` 和 `sleeves`。前两者的视觉检查通过，`sleeves` 表现不稳定。代理实验说明这个后端在所测区域上的结果，不能承担原始最优结果的复现结论。

当前方法没有使用 CNN 自动锚点或显式 Itô / MSE 理论。确定性保证来自掩码外权重为零与最终像素恢复；范围内的语义与形体变化来自编辑器生成的候选图。更换候选图或掩码会改变范围内结果，审计通过也不等于审美成功。
