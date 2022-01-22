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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Converts a SINGLE input image Tensor into a TensorList of pixels, including coordinate and color values
 */
@SuppressWarnings("serial")
public class CoordinateDisassemblyLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CoordinateDisassemblyLayer.class);
  private final boolean outputColors;
  private double minX = 0.0;
  private double minY = 0.0;
  //  private double minZ = 0.0;
  private double maxX = 1.0;
  private double maxY = 1.0;
//  private double maxZ = 1.0;


  public CoordinateDisassemblyLayer() {
    this(true);
  }

  /**
   * Instantiates a new Img band bias layer.
   */
  protected CoordinateDisassemblyLayer(boolean outputColors) {
    super();
    this.outputColors = outputColors;
  }


  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected CoordinateDisassemblyLayer(@Nonnull final JsonObject json) {
    super(json);
    minX = (json.getAsJsonPrimitive("minX")).getAsDouble();
    minY = (json.getAsJsonPrimitive("minY")).getAsDouble();
//    minZ = (json.getAsJsonPrimitive("minZ")).getAsDouble();
    maxX = (json.getAsJsonPrimitive("maxX")).getAsDouble();
    maxY = (json.getAsJsonPrimitive("maxY")).getAsDouble();
    outputColors = (json.getAsJsonPrimitive("outputColors")).getAsBoolean();
//    maxZ = (json.getAsJsonPrimitive("maxZ")).getAsDouble();
  }

  /**
   * From json img band bias layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band bias layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static CoordinateDisassemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CoordinateDisassemblyLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    Result result = eval(inObj[0].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  /**
   * Eval result.
   *
   * @param input the input
   * @return the result
   */
  @Nonnull
  public Result eval(@Nonnull final Result input) {
    try {
      TensorList inputData = input.getData();
      @Nonnull int[] inputDimensions = inputData.getDimensions();
      TensorArray data = fwd(inputDimensions, inputData);
      boolean alive = input.isAlive();
      Accumulator accumulator = new Accumulator(inputDimensions, input.getAccumulator(), alive, outputColors);
      return new Result(data, accumulator, alive || !isFrozen());
    } finally {
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("minX", minX);
    json.addProperty("minY", minY);
//    json.addProperty("minZ", minZ);
    json.addProperty("maxX", maxX);
    json.addProperty("maxY", maxY);
    json.addProperty("outputColors", outputColors);
//    json.addProperty("maxZ", maxZ);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefList.empty();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  CoordinateDisassemblyLayer addRef() {
    return (CoordinateDisassemblyLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(@Nonnull int[] inputDimensions, TensorList inputData) {
    if ((inputData.length() != 1)) {
      inputData.freeRef();
      throw new IllegalArgumentException();
    }
    Tensor image = inputData.get(0);
    inputData.freeRef();
    int[] dimensions = image.getDimensions();
    try {
      if (!Arrays.equals(dimensions, inputDimensions)) {
        throw new IllegalArgumentException(String.format("Tensor dims %s does not match TensorList dims %s", Arrays.toString(dimensions), Arrays.toString(inputDimensions)));
      }
      Tensor[] pixels = RefIntStream.range(0, dimensions[1]).mapToObj(x -> x).flatMap(y ->
          RefIntStream.range(0, dimensions[0]).mapToObj(x -> {
            Tensor tensor = new Tensor(outputColors ? (2 + dimensions[2]) : 2);
            try {
              return tensor.mapIndex((v, i) -> {
                if (i == 0) {
                  return ((double) x / dimensions[0]) * (maxX - minX) + minX;
                } else if (i == 1) {
                  return ((double) y / dimensions[1]) * (maxY - minY) + minY;
                } else {
                  return outputColors ? image.get(x, y, i - 2) : 0;
                }
              });
            } finally {
              tensor.freeRef();
            }
          })).toArray(Tensor[]::new);
      return new TensorArray(pixels);
    } finally {
      image.freeRef();
    }
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] dimensions;
    private final boolean outputColors;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param dimensions         input resolution
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(int[] dimensions, Result.Accumulator accumulator, boolean alive, boolean outputColors) {
      this.dimensions = dimensions;
      this.accumulator = accumulator;
      this.alive = alive;
      this.outputColors = outputColors;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        Tensor coordImage = new Tensor(this.dimensions[0], this.dimensions[1]);
        int coordOffset = outputColors ? 2 : 0;
        Tensor tensor;
        if(outputColors) {
          tensor = new Tensor(IntStream.range(0, this.dimensions[2]).mapToObj(x->x)
              .flatMapToDouble(z -> DoubleStream.of(coordImage.coordStream(false).mapToDouble(c ->
                  data.get(c.getIndex(), coordOffset + z)).toArray())).toArray(), this.dimensions);
        } else {
          tensor = new Tensor(this.dimensions);
        }

        coordImage.freeRef();
        data.freeRef();
        accumulator.accept(buffer, new TensorArray(tensor));
      } else {
        data.freeRef();
        buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
