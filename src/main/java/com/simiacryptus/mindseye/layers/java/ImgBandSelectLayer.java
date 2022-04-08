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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

/**
 * This class is responsible for selecting the bands to be used in an image.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class ImgBandSelectLayer extends LayerBase {

  private final int[] bands;

  /**
   * Instantiates a new Img band select layer.
   *
   * @param bands the bands
   */
  public ImgBandSelectLayer(final int... bands) {
    super();
    this.bands = bands;
  }

  /**
   * Instantiates a new Img band select layer.
   *
   * @param json the json
   */
  protected ImgBandSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray jsonArray = json.getAsJsonArray("bands");
    bands = new int[jsonArray.size()];
    for (int i = 0; i < bands.length; i++) {
      bands[i] = jsonArray.get(i).getAsInt();
    }
  }

  /**
   * Creates a new {@link ImgBandSelectLayer} from the given JSON object.
   *
   * @param json the JSON object to create the layer from
   * @param rs   the map of raw data sources
   * @return the new {@link ImgBandSelectLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull TensorArray wrap = fwd(batch, inputDims);
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(bands, inputDims, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(wrap, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray array = new JsonArray();
    for (final int b : bands) {
      array.add(new JsonPrimitive(b));
    }
    json.add("bands", array);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
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
  ImgBandSelectLayer addRef() {
    return (ImgBandSelectLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList batch, int[] inputDims) {
    @Nonnull final Tensor outputDims = new Tensor(inputDims[0], inputDims[1], bands.length);
    return new TensorArray(RefIntStream.range(0, batch.length()).parallel().mapToObj(RefUtil
            .wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              Tensor tensor = batch.get(dataIndex);
              return outputDims.mapCoords(RefUtil.wrapInterface(c -> {
                int[] coords = c.getCoords();
                return tensor.get(coords[0], coords[1], bands[coords[2]]);
              }, batch.addRef(), tensor));
            }, outputDims, batch))
        .toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to hold input dimensions, bands, and an accumulator.
   * This class also has a boolean to check if it is alive.
   *
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private int[] bands;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param bands       the bands
     * @param inputDims   the input dims
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(int[] bands, int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.inputDims = inputDims;
      this.bands = bands;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              err.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                int[] coords = c.getCoords();
                passback.set(coords[0], coords[1], bands[coords[2]], err.get(c));
              }, passback.addRef(), err));
              return passback;
            }, error)).toArray(Tensor[]::new));
        this.accumulator.accept(buffer, tensorArray);
      } else {
        error.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
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
