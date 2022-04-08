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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * This class represents a ReshapeLayer.
 *
 * @author AuthorName
 * @version 1.0
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class ReshapeLayer extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(ReshapeLayer.class);
  /**
   * The Output dims.
   */
  @Nullable
  public final int[] outputDims;

  private ReshapeLayer() {
    outputDims = null;
  }

  /**
   * Instantiates a new Reshape layer.
   *
   * @param outputDims the output dims
   */
  public ReshapeLayer(@Nonnull final int... outputDims) {
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
  }

  /**
   * Instantiates a new Reshape layer.
   *
   * @param json the json
   */
  protected ReshapeLayer(@Nonnull final JsonObject json) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
  }

  /**
   * Creates a new {@link ReshapeLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link ReshapeLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ReshapeLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ReshapeLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result in0 = inObj[0].addRef();
    TensorList data0 = in0.getData();
    RefUtil.freeRef(inObj);
    @Nonnull
    int[] inputDims = data0.getDimensions();
    ReshapedTensorList data = new ReshapedTensorList(data0, outputDims);
    boolean alive = in0.isAlive();
    Accumulator accumulator = new Accumulator(inputDims, in0.getAccumulator());
    in0.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert outputDims != null;
    json.add("outputDims", JsonUtil.getJson(outputDims));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ReshapeLayer addRef() {
    return (ReshapeLayer) super.addRef();
  }

  /**
   * The Accumulator class is used to hold an array of input dimensions and a Result.Accumulator object.
   *
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputDims   the input dims
     * @param accumulator the accumulator
     */
    public Accumulator(int[] inputDims, Result.Accumulator accumulator) {
      this.inputDims = inputDims;
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      this.accumulator.accept(buffer, new ReshapedTensorList(delta, inputDims));
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
