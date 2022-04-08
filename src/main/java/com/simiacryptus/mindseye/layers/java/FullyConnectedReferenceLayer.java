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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.*;

/**
 * This class represents a fully connected reference layer.
 *
 * @param inputDims  the input dimensions
 * @param outputDims the output dimensions
 * @param weights    the weights
 * @author John Doe
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class FullyConnectedReferenceLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedReferenceLayer.class);
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
  /**
   * The Weights.
   */
  @Nullable
  public final Tensor weights;

  /**
   * Instantiates a new Fully connected reference layer.
   */
  protected FullyConnectedReferenceLayer() {
    super();
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  /**
   * Instantiates a new Fully connected reference layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedReferenceLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int inputs = Tensor.length(inputDims);
    final int outputs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outputs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outputs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  /**
   * Instantiates a new Fully connected reference layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected FullyConnectedReferenceLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
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
   * Sets the weights of the map by iterating through each coordinate and applying the given function.
   *
   * @param f the function to apply to each coordinate
   * @docgenVersion 9
   */
  public void setByCoord(@Nonnull ToDoubleFunction<Coordinate> f) {
    assert weights != null;
    weights.coordStream(true).forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
  }

  /**
   * Sets the value of this coordinate by using a function that takes in
   * two coordinates and outputs a double.
   *
   * @param f the function to use
   * @throws NullPointerException if f is null
   * @docgenVersion 9
   */
  public void setByCoord(@Nonnull ToDoubleBiFunction<Coordinate, Coordinate> f) {
    assert inputDims != null;
    Tensor temp_02_0008 = new Tensor(inputDims);
    temp_02_0008.coordStream(true).forEach(in -> {
      assert outputDims != null;
      Tensor temp_02_0009 = new Tensor(outputDims);
      temp_02_0009.coordStream(true).forEach(out -> {
        assert weights != null;
        weights.set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
      });
      temp_02_0009.freeRef();
    });
    temp_02_0008.freeRef();
  }

  /**
   * Sets the weights of the log to the given value.
   *
   * @param value the value to set the weights to
   * @docgenVersion 9
   */
  public void setWeightsLog(double value) {
    assert weights != null;
    weights.coordStream(false).forEach(c -> {
      weights.set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    });
  }

  /**
   * Creates a new {@link FullyConnectedReferenceLayer} from the given JSON object.
   *
   * @param json the JSON object to create the layer from
   * @param rs   the map of character sequences to byte arrays
   * @return the newly created layer
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static FullyConnectedReferenceLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedReferenceLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result inputResult = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList indata = inputResult.getData();
    @Nonnull
    int[] inputDimensions = indata.getDimensions();
    assert this.inputDims != null;
    assert Tensor.length(inputDimensions) == Tensor.length(this.inputDims) : RefArrays
        .toString(inputDimensions) + " == " + RefArrays.toString(this.inputDims);
    boolean alive = inputResult.isAlive();
    Result.Accumulator accumulator = new Accumulator(indata.addRef(), getWeights(), getId(), inputDims, isFrozen(), inputResult.getAccumulator(), inputResult.isAlive());
    inputResult.freeRef();
    TensorArray data = fwd(indata);
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert outputDims != null;
    json.add("outputDims", JsonUtil.getJson(outputDims));
    assert inputDims != null;
    json.add("inputDims", JsonUtil.getJson(inputDims));
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  /**
   * Sets the value of each element in the array to the result of the given DoubleSupplier.
   *
   * @param f the DoubleSupplier used to set the value of each element
   * @docgenVersion 9
   */
  public void set(@Nonnull final DoubleSupplier f) {
    assert weights != null;
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
  }

  /**
   * Sets the weights of the perceptron using the given function.
   *
   * @param f the function to use for setting the weights
   * @docgenVersion 9
   */
  public void set(@Nonnull IntToDoubleFunction f) {
    assert weights != null;
    weights.set(f);
  }

  /**
   * Sets the weights of the perceptron.
   *
   * @param data the new weights of the perceptron
   * @docgenVersion 9
   */
  public void set(double[] data) {
    assert weights != null;
    weights.set(data);
  }

  /**
   * Sets the weights of the layer to the given data.
   *
   * @param data the new weights for the layer
   * @docgenVersion 9
   */
  public void set(@Nonnull Tensor data) {
    assert weights != null;
    weights.set(data);
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor weights = getWeights();
    assert weights != null;
    RefList<double[]> temp_02_0011 = RefArrays.asList(weights.getData());
    weights.freeRef();
    return temp_02_0011;
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
  FullyConnectedReferenceLayer addRef() {
    return (FullyConnectedReferenceLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList indata) {
    return new TensorArray(RefIntStream.range(0, indata.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
          @Nullable final Tensor input = indata.get(index);
          assert outputDims != null;
          @Nullable final Tensor output = new Tensor(outputDims);
          assert weights != null;
          weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
            int[] coords = c.getCoords();
            double prev = output.get(coords[1]);
            double w = weights.get(c);
            double i = input.get(coords[0]);
            double value = prev + w * i;
            output.set(coords[1], value);
          }, input.addRef(), output.addRef()));
          input.freeRef();
          return output;
        }, indata)).toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to accumulate the input data and weights.
   *
   * @param indata      The input data to be accumulated.
   * @param weights     The weights to be accumulated.
   * @param id          The id of the Accumulator.
   * @param frozen      Whether the Accumulator is frozen.
   * @param inputDims   The input dimensions of the Accumulator.
   * @param accumulator The accumulator of the Accumulator.
   * @param alive       Whether the Accumulator is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private Tensor weights;
    private UUID id;
    private boolean frozen;
    private int[] inputDims;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param indata      the indata
     * @param weights     the weights
     * @param id          the id
     * @param inputDims   the input dims
     * @param frozen      the frozen
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(TensorList indata, Tensor weights, UUID id, int[] inputDims, boolean frozen, Result.Accumulator accumulator, boolean alive) {
      this.indata = indata;
      this.weights = weights;
      this.id = id;
      this.frozen = frozen;
      this.inputDims = inputDims;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (!frozen) {
        Tensor[] array = RefIntStream.range(0, indata.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                  @Nullable final Tensor inputTensor = indata.get(i);
                  @Nullable final Tensor deltaTensor = delta.get(i);
                  assert weights != null;
                  @Nonnull
                  Tensor weights = new Tensor(this.weights.getDimensions());
                  weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                        int[] coords = c.getCoords();
                        weights.set(c, inputTensor.get(coords[0]) * deltaTensor.get(coords[1]));
                      }, weights.addRef(),
                      inputTensor.addRef(),
                      deltaTensor.addRef()));
                  deltaTensor.freeRef();
                  inputTensor.freeRef();
                  return weights;
                }, indata.addRef(),
                weights.addRef(),
                delta.addRef()))
            .toArray(Tensor[]::new);
        Tensor tensor = RefUtil.get(RefArrays.stream(RefUtil.addRef(array)).reduce((a, b) -> {
          return Tensor.add(a, b);
        }));
        RefUtil.freeRef(array);
        assert weights != null;
        Delta<UUID> temp_02_0010 = buffer.get(id, weights.getData());
        assert temp_02_0010 != null;
        temp_02_0010.addInPlace(tensor.getData());
        temp_02_0010.freeRef();
        tensor.freeRef();
      }
      if (alive) {
        @Nonnull final TensorList tensorList = new TensorArray(RefIntStream.range(0, indata.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
              assert inputDims != null;
              @Nullable final Tensor inputTensor = new Tensor(inputDims);
              @Nullable final Tensor deltaTensor = delta.get(i);
              assert weights != null;
              weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                    int[] coords = c.getCoords();
                    inputTensor.set(coords[0], inputTensor.get(coords[0]) + weights.get(c) * deltaTensor.get(coords[1]));
                  }, inputTensor.addRef(),
                  deltaTensor.addRef()));
              deltaTensor.freeRef();
              return inputTensor;
            }, delta.addRef())).toArray(Tensor[]::new));
        this.accumulator.accept(buffer.addRef(), tensorList);
      }
      delta.freeRef();
      buffer.freeRef();
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      weights.freeRef();
      accumulator.freeRef();
      indata.freeRef();
    }
  }
}
