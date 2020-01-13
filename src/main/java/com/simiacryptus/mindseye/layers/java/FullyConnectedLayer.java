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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.function.*;
import java.util.stream.Stream;

@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase {

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
    Tensor temp_15_0001 = null;
    weights = temp_15_0001 == null ? null : temp_15_0001.addRef();
    if (null != temp_15_0001)
      temp_15_0001.freeRef();
    inputDims = null;
  }

  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    Tensor temp_15_0002 = new Tensor(inputs, outs);
    weights = temp_15_0002 == null ? null : temp_15_0002.addRef();
    if (null != temp_15_0002)
      temp_15_0002.freeRef();
    RefUtil.freeRef(set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    }));
  }

  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    Tensor temp_15_0003 = Tensor.fromJson(json.get("weights"), resources);
    weights = temp_15_0003 == null ? null : temp_15_0003.addRef();
    if (null != temp_15_0003)
      temp_15_0003.freeRef();
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
    return weights == null ? null : weights.addRef();
  }

  @Nonnull
  public FullyConnectedLayer setByCoord(@Nonnull final ToDoubleFunction<Coordinate> f) {
    Tensor temp_15_0013 = getWeights();
    temp_15_0013.coordStream(true).forEach(c -> {
      Tensor temp_15_0014 = getWeights();
      RefUtil.freeRef(temp_15_0014.set(c, f.applyAsDouble(c)));
      if (null != temp_15_0014)
        temp_15_0014.freeRef();
    });
    if (null != temp_15_0013)
      temp_15_0013.freeRef();
    return this.addRef();
  }

  @Nonnull
  public void setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    Tensor temp_15_0011 = new Tensor(inputDims);
    temp_15_0011.coordStream(true).forEach(in -> {
      Tensor temp_15_0012 = new Tensor(outputDims);
      temp_15_0012.coordStream(true).forEach(out -> {
        Tensor temp_15_0015 = getWeights();
        temp_15_0015.set(new int[] { in.getIndex(), out.getIndex() }, f.applyAsDouble(in, out));
        if (null != temp_15_0015)
          temp_15_0015.freeRef();
      });
      if (null != temp_15_0012)
        temp_15_0012.freeRef();
    });
    if (null != temp_15_0011)
      temp_15_0011.freeRef();
  }

  @Nonnull
  public FullyConnectedLayer setWeightsLog(final double value) {
    Tensor temp_15_0016 = getWeights();
    temp_15_0016.coordStream(false).forEach(c -> {
      Tensor temp_15_0017 = getWeights();
      RefUtil.freeRef(temp_15_0017.set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value)));
      if (null != temp_15_0017)
        temp_15_0017.freeRef();
    });
    if (null != temp_15_0016)
      temp_15_0016.freeRef();
    return this.addRef();
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
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  public static void multiply(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull
    final DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }

  public static void multiplyT(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(in.length, out.length, matrix);
    @Nonnull
    final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }

  @Nonnull
  public static DoubleMatrix transpose(@Nonnull final DoubleMatrix doubleMatrix) {
    @Nonnull
    final DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows,
        RecycleBin.DOUBLES.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }

  public static @SuppressWarnings("unused") FullyConnectedLayer[] addRefs(FullyConnectedLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayer::addRef)
        .toArray((x) -> new FullyConnectedLayer[x]);
  }

  public static @SuppressWarnings("unused") FullyConnectedLayer[][] addRefs(FullyConnectedLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedLayer::addRefs)
        .toArray((x) -> new FullyConnectedLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    final FullyConnectedLayer fullyConnectedLayer = this.addRef();
    assert Tensor.length(indata.getDimensions()) == Tensor.length(fullyConnectedLayer.inputDims) : RefArrays
        .toString(indata.getDimensions()) + " == " + RefArrays.toString(fullyConnectedLayer.inputDims);
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(Tensor.length(indata.getDimensions()), Tensor.length(outputDims),
        fullyConnectedLayer.weights.getData());
    @Nonnull
    final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    @Nonnull
    TensorArray tensorArray = new TensorArray(RefIntStream.range(0, indata.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable
          final Tensor input = indata.get(dataIndex);
          @Nullable
          final Tensor output = new Tensor(outputDims);
          matrixObj.mmuli(new DoubleMatrix(input.length(), 1, input.getData()),
              new DoubleMatrix(output.length(), 1, output.getData()));
          if (null != input)
            input.freeRef();
          return output;
        }, indata == null ? null : indata.addRef())).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
    try {
      try {
        try {
          try {
            return new Result(tensorArray, new Result.Accumulator() {
              {
                Result.addRefs(inObj);
                fullyConnectedLayer.addRef();
                indata.addRef();
              }

              @Override
              public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                if (!FullyConnectedLayer.this.isFrozen()) {
                  final Delta<UUID> deltaBuffer = buffer.get(fullyConnectedLayer.getId(),
                      fullyConnectedLayer.weights.getData());
                  final int threads = 4;
                  Optional<Tensor> temp_15_0018 = RefIntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(
                      RefUtil.wrapInterface((Function<? super Integer, ? extends Stream<? extends Tensor>>) thread -> {
                        return RefIntStream.range(0, indata.length()).filter(i -> thread == i % threads)
                            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                              @Nonnull
                              final Tensor weightDelta = new Tensor(Tensor.length(inputDims),
                                  Tensor.length(outputDims));
                              Tensor deltaTensor = delta.get(dataIndex);
                              Tensor inputTensor = indata.get(dataIndex);
                              FullyConnectedLayer.crossMultiplyT(deltaTensor.getData(), inputTensor.getData(),
                                  weightDelta.getData());
                              if (null != inputTensor)
                                inputTensor.freeRef();
                              if (null != deltaTensor)
                                deltaTensor.freeRef();
                              return weightDelta;
                            }, indata == null ? null : indata.addRef(), delta == null ? null : delta.addRef()));
                      }, indata == null ? null : indata.addRef(), delta == null ? null : delta.addRef()))
                      .reduce((a, b) -> {
                        Tensor temp_15_0007 = a.addAndFree(b == null ? null : b.addRef());
                        if (null != b)
                          b.freeRef();
                        if (null != a)
                          a.freeRef();
                        return temp_15_0007;
                      });
                  RefUtil.freeRef(
                      temp_15_0018.map(RefUtil.wrapInterface((Function<? super Tensor, ? extends Delta<UUID>>) data -> {
                        Delta<UUID> temp_15_0008 = deltaBuffer.addInPlace(data.getData());
                        if (null != data)
                          data.freeRef();
                        return temp_15_0008;
                      }, deltaBuffer == null ? null : deltaBuffer.addRef())));
                  if (null != temp_15_0018)
                    RefUtil.freeRef(temp_15_0018);
                  if (null != deltaBuffer)
                    deltaBuffer.freeRef();
                }
                if (inObj[0].isAlive()) {
                  @Nonnull
                  final TensorList tensorList = new TensorArray(RefIntStream.range(0, indata.length()).parallel()
                      .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                        Tensor deltaTensor = delta.get(dataIndex);
                        @Nonnull
                        final Tensor passback = new Tensor(indata.getDimensions());
                        FullyConnectedLayer.multiply(fullyConnectedLayer.weights.getData(), deltaTensor.getData(),
                            passback.getData());
                        if (null != deltaTensor)
                          deltaTensor.freeRef();
                        return passback;
                      }, fullyConnectedLayer == null ? null : fullyConnectedLayer.addRef(),
                          indata == null ? null : indata.addRef(), delta == null ? null : delta.addRef()))
                      .toArray(i -> new Tensor[i]));
                  inObj[0].accumulate(buffer == null ? null : buffer.addRef(), tensorList == null ? null : tensorList);
                }
                if (null != delta)
                  delta.freeRef();
                if (null != buffer)
                  buffer.freeRef();
              }

              public @SuppressWarnings("unused") void _free() {
                ReferenceCounting.freeRefs(inObj);
                fullyConnectedLayer.freeRef();
                indata.freeRef();
              }
            }) {

              {
                Result.addRefs(inObj);
              }

              @Override
              public boolean isAlive() {
                return !isFrozen() || RefArrays.stream(Result.addRefs(inObj)).anyMatch(x -> {
                  boolean temp_15_0009 = x.isAlive();
                  if (null != x)
                    x.freeRef();
                  return temp_15_0009;
                });
              }

              public void _free() {
                ReferenceCounting.freeRefs(inObj);
              }

            };
          } finally {
            ReferenceCounting.freeRefs(inObj);
          }
        } finally {
          tensorArray.freeRef();
        }
      } finally {
        if (null != fullyConnectedLayer)
          fullyConnectedLayer.freeRef();
      }
    } finally {
      if (null != indata)
        indata.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    Tensor temp_15_0019 = getWeights();
    json.add("weights", temp_15_0019.getJson(resources, dataSerializer));
    if (null != temp_15_0019)
      temp_15_0019.freeRef();
    return json;
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final DoubleSupplier f) {
    Tensor temp_15_0020 = getWeights();
    RefArrays.parallelSetAll(temp_15_0020.getData(), i -> f.getAsDouble());
    if (null != temp_15_0020)
      temp_15_0020.freeRef();
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final IntToDoubleFunction f) {
    Tensor temp_15_0021 = getWeights();
    RefUtil.freeRef(temp_15_0021.set(f));
    if (null != temp_15_0021)
      temp_15_0021.freeRef();
    return this.addRef();
  }

  public void initSpacial(final double radius, final double stiffness, final double peak) {
    setByCoord((@Nonnull final Coordinate in, @Nonnull final Coordinate out) -> {
      final double[] doubleCoords = RefIntStream.range(0, in.getCoords().length).mapToDouble(d -> {
        final double from = in.getCoords()[d] * 1.0 / FullyConnectedLayer.this.inputDims[d];
        final double to = out.getCoords()[d] * 1.0 / FullyConnectedLayer.this.outputDims[d];
        return from - to;
      }).toArray();
      final double dist = Math.sqrt(RefArrays.stream(doubleCoords).map(x -> x * x).sum());
      final double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }

  @Nonnull
  public FullyConnectedLayer set(final double[] data) {
    Tensor temp_15_0022 = getWeights();
    RefUtil.freeRef(temp_15_0022.set(data));
    if (null != temp_15_0022)
      temp_15_0022.freeRef();
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedLayer set(@Nonnull final Tensor data) {
    Tensor temp_15_0023 = getWeights();
    temp_15_0023.set(data == null ? null : data);
    if (null != temp_15_0023)
      temp_15_0023.freeRef();
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedLayer scale(final double value) {
    Tensor temp_15_0024 = getWeights();
    RefUtil.freeRef(temp_15_0024.scaleInPlace(value));
    if (null != temp_15_0024)
      temp_15_0024.freeRef();
    return this.addRef();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor temp_15_0026 = getWeights();
    RefList<double[]> temp_15_0025 = RefArrays.asList(temp_15_0026.getData());
    if (null != temp_15_0026)
      temp_15_0026.freeRef();
    return temp_15_0025;
  }

  public FullyConnectedLayer randomize(double amplitude) {
    Tensor temp_15_0027 = getWeights();
    RefUtil.freeRef(temp_15_0027.randomize(amplitude));
    if (null != temp_15_0027)
      temp_15_0027.freeRef();
    return this.addRef();
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  public @Override @SuppressWarnings("unused") FullyConnectedLayer addRef() {
    return (FullyConnectedLayer) super.addRef();
  }
}
