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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jblas.DoubleMatrix;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.function.*;
import java.util.stream.Stream;

/**
 * This class represents a fully connected layer.
 *
 * @param inputDims  the input dimensions
 * @param outputDims the output dimensions
 * @param weights    the weights
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  /**
   * The Input dims.
   */
  @Nullable
  public final int[] inputDims;
  /**
   * The Output dims.
   */
  @Nullable
  public final int[] outputDims;
  @Nullable
  private final Tensor weights;

  /**
   * Instantiates a new Fully connected layer.
   */
  protected FullyConnectedLayer() {
    super();
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  /**
   * Instantiates a new Fully connected layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  /**
   * Instantiates a new Fully connected layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  /**
   * Returns the transpose of this layer.
   *
   * @return the transpose of this layer
   * @throws RuntimeException if not implemented
   * @docgenVersion 9
   */
  @Nonnull
  public Layer getTranspose() {
    throw new RuntimeException("Not Implemented");
  }

  /**
   * @return the weights, or null if they have not been set
   * @docgenVersion 9
   */
  @Nullable
  public Tensor getWeights() {
    return weights == null ? null : weights.addRef();
  }

  /**
   * Sets the weights of this layer using the given function. The function is applied to each coordinate of the weight tensor.
   *
   * @param f the function to apply to each coordinate
   * @docgenVersion 9
   */
  public void setByCoord(@Nonnull ToDoubleFunction<Coordinate> f) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.coordStream(true).forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
    weights.freeRef();
  }

  /**
   * Sets the value of this coordinate by using a function that takes in
   * two coordinates and outputs a double.
   *
   * @param f the function to use
   * @throws NullPointerException if f is null
   * @docgenVersion 9
   */
  public void setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    assert inputDims != null;
    Tensor temp_15_0011 = new Tensor(inputDims);
    temp_15_0011.coordStream(true).forEach(in -> {
      assert outputDims != null;
      Tensor weights = getWeights();
      assert weights != null;
      Tensor temp_15_0012 = new Tensor(outputDims);
      temp_15_0012.coordStream(true).forEach(out -> {
        weights.set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
      });
      temp_15_0012.freeRef();
      weights.freeRef();
    });
    temp_15_0011.freeRef();
  }

  /**
   * Sets the weightsLog.
   *
   * @param value the new weightsLog
   * @docgenVersion 9
   */
  public void setWeightsLog(double value) {
    Tensor temp_15_0016 = getWeights();
    assert temp_15_0016 != null;
    Tensor weights = getWeights();
    temp_15_0016.coordStream(false).forEach(c -> {
      weights.set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    });
    weights.freeRef();
    temp_15_0016.freeRef();
  }

  /**
   * Cross multiplies two arrays and stores the result in a third array.
   *
   * @param rows   the array of row values
   * @param cols   the array of column values
   * @param matrix the array to store the cross multiplication results
   * @docgenVersion 9
   */
  public static void crossMultiply(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }

  /**
   * Multiplies two arrays element-wise and stores the result in a third array.
   *
   * @param rows   the first array to multiply
   * @param cols   the second array to multiply
   * @param matrix the array to store the multiplication result
   * @docgenVersion 9
   */
  public static void crossMultiplyT(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double r : rows) {
      for (final double c : cols) {
        matrix[i++] = r * c;
      }
    }
  }

  /**
   * Creates a new {@link FullyConnectedLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link FullyConnectedLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }

  /**
   * Multiplies a matrix by a vector.
   *
   * @param matrix The matrix to multiply.
   * @param in     The vector to multiply.
   * @param out    The vector to store the result.
   * @docgenVersion 9
   */
  public static void multiply(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull final DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }

  /**
   * Multiplies the matrix by the vector.
   *
   * @param matrix the matrix to multiply
   * @param in     the vector to multiply
   * @param out    the vector to store the result in
   * @docgenVersion 9
   */
  public static void multiplyT(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(in.length, out.length, matrix);
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }

  /**
   * Returns the transpose of the given matrix.
   *
   * @param doubleMatrix the matrix to transpose
   * @return the transpose of the given matrix
   * @docgenVersion 9
   */
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

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    assert this.inputDims != null;
    assert Tensor.length(indata.getDimensions()) == Tensor.length(this.inputDims) : RefArrays
        .toString(indata.getDimensions()) + " == " + RefArrays.toString(this.inputDims);
    assert this.weights != null;
    assert outputDims != null;
    @Nonnull TensorArray data = fwd(indata.addRef());
    final Result.Accumulator accumulator1 = inObj[0].getAccumulator();
    final boolean alive1 = inObj[0].isAlive();
    Accumulator accumulator = new Accumulator(indata, inputDims, outputDims, isFrozen(), this.getId(), this.weights.addRef(), accumulator1, alive1);
    boolean alive = RefArrays.stream(inObj).anyMatch(x -> {
      boolean xAlive = x.isAlive();
      x.freeRef();
      return xAlive;
    });
    return new Result(data, accumulator, !isFrozen() || alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert outputDims != null;
    json.add("outputDims", JsonUtil.getJson(outputDims));
    assert inputDims != null;
    json.add("inputDims", JsonUtil.getJson(inputDims));
    Tensor weights = getWeights();
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    weights.freeRef();
    return json;
  }

  /**
   * Sets the weights of this layer using the given {@link DoubleSupplier}.
   *
   * @param f the {@link DoubleSupplier} to use for setting weights
   * @docgenVersion 9
   */
  public void set(@Nonnull DoubleSupplier f) {
    Tensor weights = getWeights();
    assert weights != null;
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    weights.freeRef();
  }

  /**
   * Sets the weights of this layer using the given function.
   *
   * @param f the function to use for setting weights
   * @docgenVersion 9
   */
  public void set(@Nonnull IntToDoubleFunction f) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.set(f);
    weights.freeRef();
  }

  /**
   * Initializes the spacial parameters for the simulation.
   *
   * @param radius    the radius of the spacial area
   * @param stiffness the stiffness of the spacial area
   * @param peak      the peak value of the spacial area
   * @docgenVersion 9
   */
  public void initSpacial(final double radius, final double stiffness, final double peak) {
    setByCoord((@Nonnull final Coordinate in, @Nonnull final Coordinate out) -> {
      final double[] doubleCoords = RefIntStream.range(0, in.getCoords().length).mapToDouble(d -> {
        assert inputDims != null;
        final double from = in.getCoords()[d] * 1.0 / inputDims[d];
        assert outputDims != null;
        final double to = out.getCoords()[d] * 1.0 / outputDims[d];
        return from - to;
      }).toArray();
      final double dist = Math.sqrt(RefArrays.stream(doubleCoords).map(x -> x * x).sum());
      final double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }

  /**
   * Sets the weights of this layer.
   *
   * @param data The weights to set.
   * @docgenVersion 9
   */
  public void set(double[] data) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.set(data);
    weights.freeRef();
  }

  /**
   * Sets the weights of this layer.
   *
   * @param data The weights to set.
   * @docgenVersion 9
   */
  public void set(@Nonnull Tensor data) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.set(data);
    weights.freeRef();
  }

  /**
   * Scales the weights of this layer by the specified value.
   *
   * @param value the value to scale the weights by
   * @docgenVersion 9
   */
  public void scale(double value) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.scaleInPlace(value);
    weights.freeRef();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor weights = getWeights();
    assert weights != null;
    RefList<double[]> temp_15_0025 = RefArrays.asList(weights.getData());
    weights.freeRef();
    return temp_15_0025;
  }

  /**
   * Randomizes the weights of the layer with a given amplitude.
   *
   * @param amplitude the amplitude
   * @docgenVersion 9
   */
  public void randomize(double amplitude) {
    Tensor weights = getWeights();
    assert weights != null;
    weights.randomize(amplitude);
    weights.freeRef();
  }

  /**
   * Frees resources used by this object.
   *
   * @docgenVersion 9
   */
  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  FullyConnectedLayer addRef() {
    return (FullyConnectedLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList indata) {
    @Nonnull
    DoubleMatrix doubleMatrix = new DoubleMatrix(Tensor.length(indata.getDimensions()), Tensor.length(outputDims),
        this.weights.getData());
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    @Nonnull
    TensorArray tensorArray = new TensorArray(RefIntStream.range(0, indata.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable final Tensor input = indata.get(dataIndex);
          @Nullable final Tensor output = new Tensor(outputDims);
          matrixObj.mmuli(new DoubleMatrix(input.length(), 1, input.getData()),
              new DoubleMatrix(output.length(), 1, output.getData()));
          input.freeRef();
          return output;
        }, indata)).toArray(Tensor[]::new));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
    return tensorArray;
  }

  /**
   * The Accumulator class is used to accumulate the input data.
   *
   * @param indata      The input data to be accumulated.
   * @param frozen      Whether or not the input data is frozen.
   * @param inputDims   The input dimensions of the data.
   * @param outputDims  The output dimensions of the data.
   * @param id          The id of the data.
   * @param weights     The weights of the data.
   * @param accumulator The accumulator of the data.
   * @param alive       Whether or not the data is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private boolean frozen;
    private int[] inputDims;
    private int[] outputDims;
    private UUID id;
    private Tensor weights;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param indata      the indata
     * @param inputDims   the input dims
     * @param outputDims  the output dims
     * @param frozen      the frozen
     * @param id          the id
     * @param weights     the weights
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(TensorList indata, int[] inputDims, int[] outputDims, boolean frozen, UUID id, Tensor weights, Result.Accumulator accumulator, boolean alive) {
      this.indata = indata;
      this.frozen = frozen;
      this.inputDims = inputDims;
      this.outputDims = outputDims;
      this.id = id;
      this.weights = weights;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (!frozen) {
        final Delta<UUID> deltaBuffer = buffer.get(id,
            weights.getData());
        final int threads = 4;
        Optional<Tensor> temp_15_0018 = RefIntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(
                RefUtil.wrapInterface((Function<? super Integer, ? extends Stream<? extends Tensor>>) thread -> {
                  return RefIntStream.range(0, indata.length()).filter(i -> thread == i % threads)
                      .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                        assert inputDims != null;
                        @Nonnull final Tensor weightDelta = new Tensor(Tensor.length(inputDims),
                            Tensor.length(outputDims));
                        Tensor deltaTensor = delta.get(dataIndex);
                        Tensor inputTensor = indata.get(dataIndex);
                        FullyConnectedLayer.crossMultiplyT(deltaTensor.getData(), inputTensor.getData(),
                            weightDelta.getData());
                        inputTensor.freeRef();
                        deltaTensor.freeRef();
                        return weightDelta;
                      }, indata.addRef(), delta.addRef()));
                }, indata.addRef(), delta.addRef()))
            .reduce((a, b) -> {
              return Tensor.add(a, b);
            });
        RefUtil.freeRef(
            RefUtil.map(temp_15_0018, RefUtil.wrapInterface((Function<Tensor, Delta<UUID>>) data -> {
              assert deltaBuffer != null;
              deltaBuffer.addInPlace(data.getData());
              Delta<UUID> temp_15_0008 = deltaBuffer.addRef();
              data.freeRef();
              return temp_15_0008;
            }, deltaBuffer == null ? null : deltaBuffer.addRef())));
        if (null != deltaBuffer)
          deltaBuffer.freeRef();
      }
      if (alive) {
        @Nonnull final TensorList tensorList = new TensorArray(RefIntStream.range(0, indata.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nonnull final Tensor passback = new Tensor(indata.getDimensions());
                  FullyConnectedLayer.multiply(weights.getData(), deltaTensor.getData(),
                      passback.getData());
                  deltaTensor.freeRef();
                  return passback;
                },
                indata.addRef(), delta.addRef()))
            .toArray(Tensor[]::new));
        this.accumulator.accept(buffer.addRef(), tensorList);
      }
      delta.freeRef();
      buffer.freeRef();
    }

    /**
     * Frees any resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(accumulator);
      weights.freeRef();
      indata.freeRef();
    }
  }
}
