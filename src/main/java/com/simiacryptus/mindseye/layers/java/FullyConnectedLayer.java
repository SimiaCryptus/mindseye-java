/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class FullyConnectedLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  @Nullable
  public final int[] inputDims;
  @Nullable
  public final int[] outputDims;
  @Nullable
  private final Tensor weights;

  protected FullyConnectedLayer() {
    super();
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = com.simiacryptus.ref.wrappers.RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = com.simiacryptus.ref.wrappers.RefArrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  protected FullyConnectedLayer(@Nonnull final JsonObject json,
                                com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  @Nonnull
  public Layer getTranspose() {
    throw new RuntimeException("Not Implemented");
  }

  /**
   * The Weights.
   */
  @Nullable
  public Tensor getWeights() {
    return weights;
  }

  @Nonnull
  public FullyConnectedLayer setByCoord(@Nonnull final ToDoubleFunction<Coordinate> f) {
    getWeights().coordStream(true).forEach(c -> {
      getWeights().set(c, f.applyAsDouble(c));
    });
    return this;
  }

  @Nonnull
  public void setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream(true).forEach(in -> {
      new Tensor(outputDims).coordStream(true).forEach(out -> {
        getWeights().set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
      });
    });
  }

  @Nonnull
  public FullyConnectedLayer setWeightsLog(final double value) {
    getWeights().coordStream(false).forEach(c -> {
      getWeights().set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }

  public static void crossMultiply(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }

  public static void crossMultiplyT(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double r : rows) {
      for (final double c : cols) {
        matrix[i++] = r * c;
      }
    }
  }

  @SuppressWarnings("unused")
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json,
                                             com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  public static void multiply(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull final DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }

  public static void multiplyT(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(in.length, out.length, matrix);
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }

  @Nonnull
  public static DoubleMatrix transpose(@Nonnull final DoubleMatrix doubleMatrix) {
    @Nonnull final DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows,
        RecycleBin.DOUBLES.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }

  public static @SuppressWarnings("unused")
  FullyConnectedLayer[] addRefs(FullyConnectedLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayer::addRef)
        .toArray((x) -> new FullyConnectedLayer[x]);
  }

  public static @SuppressWarnings("unused")
  FullyConnectedLayer[][] addRefs(FullyConnectedLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayer::addRefs)
        .toArray((x) -> new FullyConnectedLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    assert Tensor.length(indata.getDimensions()) == Tensor
        .length(this.inputDims) : com.simiacryptus.ref.wrappers.RefArrays.toString(indata.getDimensions()) + " == "
        + com.simiacryptus.ref.wrappers.RefArrays.toString(this.inputDims);
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(Tensor.length(indata.getDimensions()), Tensor.length(outputDims),
        this.weights.getData());
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    @Nonnull
    TensorArray tensorArray = new TensorArray(
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
          @Nullable final Tensor input = indata.get(dataIndex);
          @Nullable final Tensor output = new Tensor(outputDims);
          matrixObj.mmuli(new DoubleMatrix(input.length(), 1, input.getData()),
              new DoubleMatrix(output.length(), 1, output.getData()));
          return output;
        }).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
    return new Result(tensorArray, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        final Delta<UUID> deltaBuffer = buffer.get(FullyConnectedLayer.this.getId(), this.weights.getData());
        final int threads = 4;
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(thread -> {
          return com.simiacryptus.ref.wrappers.RefIntStream.range(0, indata.length()).filter(i -> thread == i % threads)
              .mapToObj(dataIndex -> {
                @Nonnull final Tensor weightDelta = new Tensor(Tensor.length(inputDims), Tensor.length(outputDims));
                Tensor deltaTensor = delta.get(dataIndex);
                Tensor inputTensor = indata.get(dataIndex);
                FullyConnectedLayer.crossMultiplyT(deltaTensor.getData(), inputTensor.getData(), weightDelta.getData());
                return weightDelta;
              });
        }).reduce((a, b) -> {
          return a.addAndFree(b);
        }).map(data -> {
          return deltaBuffer.addInPlace(data.getData());
        });
      }
      if (inObj[0].isAlive()) {
        @Nonnull final TensorList tensorList = new TensorArray(
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
              Tensor deltaTensor = delta.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(indata.getDimensions());
              FullyConnectedLayer.multiply(this.weights.getData(), deltaTensor.getData(), passback.getData());
              return passback;
            }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorList);
      }
    }) {

      @Override
      public boolean isAlive() {
        return !isFrozen() || com.simiacryptus.ref.wrappers.RefArrays.stream(inObj).anyMatch(x -> x.isAlive());
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", getWeights().getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final DoubleSupplier f) {
    com.simiacryptus.ref.wrappers.RefArrays.parallelSetAll(getWeights().getData(), i -> f.getAsDouble());
    return this;
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final IntToDoubleFunction f) {
    getWeights().set(f);
    return this;
  }

  public void initSpacial(final double radius, final double stiffness, final double peak) {
    setByCoord((@Nonnull final Coordinate in, @Nonnull final Coordinate out) -> {
      final double[] doubleCoords = com.simiacryptus.ref.wrappers.RefIntStream.range(0, in.getCoords().length)
          .mapToDouble(d -> {
            final double from = in.getCoords()[d] * 1.0 / FullyConnectedLayer.this.inputDims[d];
            final double to = out.getCoords()[d] * 1.0 / FullyConnectedLayer.this.outputDims[d];
            return from - to;
          }).toArray();
      final double dist = Math.sqrt(com.simiacryptus.ref.wrappers.RefArrays.stream(doubleCoords).map(x -> x * x).sum());
      final double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }

  @Nonnull
  public FullyConnectedLayer set(final double[] data) {
    getWeights().set(data);
    return this;
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final Tensor data) {
    getWeights().set(data);
    return this;
  }

  @Nonnull
  public FullyConnectedLayer scale(final double value) {
    getWeights().scaleInPlace(value);
    return this;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getWeights().getData());
  }

  public FullyConnectedLayer randomize(double amplitude) {
    getWeights().randomize(amplitude);
    return this;
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  FullyConnectedLayer addRef() {
    return (FullyConnectedLayer) super.addRef();
  }
}
